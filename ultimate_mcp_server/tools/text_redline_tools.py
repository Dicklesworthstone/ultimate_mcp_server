# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import datetime as _dt
import difflib
import hashlib
import html as html_stdlib
import itertools
import json
import re
import subprocess
import tempfile
import textwrap
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import markdown
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from lxml import etree
from lxml import html as lxml_html
from lxml.etree import _Element, _ElementTree
from xmldiff import formatting, main
from xmldiff.actions import (
    DeleteAttrib,
    DeleteNode,
    InsertAttrib,
    InsertNode,
    MoveNode,
    RenameAttrib,
    UpdateAttrib,
    UpdateTextIn,
)

try:
    from xmldiff.actions import InsertComment
except ImportError:
    InsertComment = None
try:
    from xmldiff.actions import RenameNode
except ImportError:
    RenameNode = None
try:
    from xmldiff.actions import UpdateTextAfter
except ImportError:
    UpdateTextAfter = None
try:
    from xmldiff.actions import UpdateTextBefore
except ImportError:
    UpdateTextBefore = None
try:
    from xmldiff.actions import (
        DeleteTextAfter,
        DeleteTextBefore,
        DeleteTextIn,
        InsertTextAfter,
        InsertTextBefore,
        InsertTextIn,
    )

    _fine_grained_text_actions = True
except ImportError:
    InsertTextIn = InsertTextBefore = InsertTextAfter = None
    DeleteTextIn = DeleteTextBefore = DeleteTextAfter = None
    _fine_grained_text_actions = False
    _INSERT_TEXT_CLS = tuple()
    _DELETE_TEXT_CLS = tuple()
    _UPDATE_TEXT_CLS = tuple(c for c in (UpdateTextIn, UpdateTextBefore, UpdateTextAfter) if c)

if _fine_grained_text_actions:
    _INSERT_TEXT_CLS = tuple(c for c in (InsertTextIn, InsertTextBefore, InsertTextAfter) if c)
    _DELETE_TEXT_CLS = tuple(c for c in (DeleteTextIn, DeleteTextBefore, DeleteTextAfter) if c)
    _UPDATE_TEXT_CLS = tuple(c for c in (UpdateTextIn, UpdateTextBefore, UpdateTextAfter) if c)
else:
    _INSERT_TEXT_CLS = _INSERT_TEXT_CLS if "_INSERT_TEXT_CLS" in locals() else tuple()
    _DELETE_TEXT_CLS = _DELETE_TEXT_CLS if "_DELETE_TEXT_CLS" in locals() else tuple()
    _UPDATE_TEXT_CLS = (
        _UPDATE_TEXT_CLS
        if "_UPDATE_TEXT_CLS" in locals()
        else tuple(c for c in (UpdateTextIn, UpdateTextBefore, UpdateTextAfter) if c)
    )

_ATTR_UPDATE_CLS = (UpdateAttrib,)
_ATTR_INSERT_CLS = (InsertAttrib,)
_ATTR_DELETE_CLS = (DeleteAttrib,)
_ATTR_RENAME_CLS = (RenameAttrib,)
# ------------------------------------------------------------------

from ultimate_mcp_server.exceptions import ToolError, ToolInputError  # noqa: E402
from ultimate_mcp_server.tools.base import with_error_handling, with_tool_metrics  # noqa: E402
from ultimate_mcp_server.utils import get_logger  # noqa: E402

# --- Document Conversion Import ---
try:
    from ultimate_mcp_server.tools.document_conversion_and_processing import convert_document

    _DOC_CONVERSION_AVAILABLE = True
except ImportError:
    convert_document = None
    _DOC_CONVERSION_AVAILABLE = False
    print("Document conversion tool not available. Non-HTML input comparison will fail.")
# ---------------------------------

logger = get_logger("ultimate_mcp_server.tools.redline")

# --- Add logger.exception if missing ---
if not hasattr(logger, "exception"):

    def _logger_exception(msg, *args, **kw):
        logger.error(msg, *args, exc_info=True, **kw)

    logger.exception = _logger_exception  # type: ignore[attr-defined]
# -----------------------------------------

# Namespace and configuration constants
_DIFF_NS = "http://namespaces.shoobx.com/diff"
_DIFF_PREFIX = "diff"

# --- Synthetic ID Generation ---
_id_counter = itertools.count(1)


def _normalize_text(text: Optional[str]) -> str:
    """Collapses whitespace and strips leading/trailing space."""
    if text is None:
        return ""
    # Replace various whitespace chars with a single space, then strip
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized


def _get_normalized_full_text(el: _Element) -> str:
    """Gets normalized text content of an element and its descendants,
    excluding script and style tags."""
    if el is None:
        return ""
    # Get text from all descendant text nodes, excluding those within script/style
    # We join with space to handle cases like <p>Text<b>bold</b> more</p>
    try:
        texts = el.xpath(".//text()[not(ancestor::script) and not(ancestor::style)]")
        full_text = " ".join(t.strip() for t in texts if t.strip())
        # Normalize the combined text
        return _normalize_text(full_text)
    except Exception as e:
        # Fallback for safety, though xpath should be robust
        logger.warning(f"XPath text extraction failed for <{el.tag}>: {e}. Falling back.")
        texts = [t for t in el.itertext() if t.strip()]  # Less precise about script/style
        full_text = " ".join(t.strip() for t in texts)
        return _normalize_text(full_text)


# Define significant attributes (adjust as needed)
# These are attributes likely to uniquely identify an element or its purpose
# Avoid volatile attributes like style, or overly common ones like class (unless very specific)
_SIGNIFICANT_ATTRIBUTES = {"id", "href", "src", "name", "value", "title", "alt", "rel", "type"}
# Consider adding data-* attributes if they are known to be stable identifiers in your source HTML

# --- Fuzzy move detection configuration ---
_MOVE_MIN_TOKENS: int = 5  # Ignore tiny fragments; raise to 8 for stricter pairing
_MOVE_MIN_CHARS: int = 24  # Shorter text tends to be noise
_MOVE_SIM_THRESHOLD: float = 0.82  # 0..1; lower to catch more, higher to be stricter
_MOVE_MAX_CANDIDATES: int = 2000  # Safety bound for O(N*M) matching
# Tags considered "blocky enough" to attempt fuzzy move pairing even if tags differ
_BLOCK_TAGS: Set[str] = {
    "p",
    "li",
    "ul",
    "ol",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "pre",
    "code",
    "section",
    "article",
    "aside",
    "figure",
    "figcaption",
    "table",
    "thead",
    "tbody",
    "tr",
    "td",
    "th",
    "dl",
    "dt",
    "dd",
}
# Attributes that help anchor identity/context
_ANCHOR_ATTRS: Set[str] = {
    "id",
    "name",
    "href",
    "src",
    "data-id",
    "data-key",
    "data-uid",
    "aria-label",
    "title",
}


def _inject_synthetic_ids(root: _Element, *, attr: str = "data-diff-id") -> None:
    """Inject synthetic IDs into elements based on tag, normalized full text,
    and significant attributes."""
    global _id_counter
    if root is None:
        return

    processed_elements = 0
    elements_with_ids = 0

    # Iterate through all elements in the tree
    for el in root.iter():
        if not isinstance(el, _Element):
            continue

        processed_elements += 1

        # Skip if ID already exists (e.g., from previous run or source)
        if el.get(attr):
            elements_with_ids += 1
            continue

        # 1. Get Tag
        tag = el.tag

        # 2. Get Normalized Full Text Content
        norm_text = _get_normalized_full_text(el)

        # 3. Get Normalized Significant Attributes
        sig_attrs = {}
        for k, v in el.attrib.items():
            # Check if attribute is considered significant OR if it's a data-* attribute
            # (often used for stable identifiers)
            # Exclude the synthetic ID attribute itself if looping
            if (k in _SIGNIFICANT_ATTRIBUTES or k.startswith("data-")) and k != attr:
                # Normalize attribute value's whitespace
                sig_attrs[k] = _normalize_text(v)

        # Sort significant attributes by key for consistent signature
        sorted_sig_attrs = tuple(sorted(sig_attrs.items()))

        # 4. Create Signature Tuple
        # Using a hash of the potentially long text to keep the signature manageable
        text_hash = hashlib.blake2b(norm_text.encode("utf-8", "replace"), digest_size=8).hexdigest()
        sig_tuple = (tag, text_hash, sorted_sig_attrs)

        # 5. Generate Hash and Synthetic ID
        try:
            # Hash the representation of the signature tuple
            sig_repr = repr(sig_tuple).encode("utf-8", "replace")
            h = hashlib.blake2b(sig_repr, digest_size=8).hexdigest()
            # Combine counter and hash for uniqueness
            synthetic_id = f"synid_{next(_id_counter):06d}_{h}"
            el.set(attr, synthetic_id)
            elements_with_ids += 1
        except Exception as e:
            logger.warning(
                f"Failed to generate/set synthetic ID for element <{el.tag}> "
                f"(Text hash: {text_hash}, Attrs: {sorted_sig_attrs}): {e}"
            )

    logger.debug(
        f"ID Injection: Processed {processed_elements} elements, {elements_with_ids} have IDs."
    )


# Helper to safely get attributes from actions
def _safe_get_attr(action: Any, *attr_names: str, default: Any = None) -> Any:
    if action is None:
        return default
    for name in attr_names:
        if "." in name:
            parts = name.split(".")
            obj = action
            try:
                for part in parts:
                    if obj is None or not hasattr(obj, part):
                        obj = None
                        break
                    obj = getattr(obj, part)
                if obj is not None:
                    return obj
            except (AttributeError, TypeError):
                continue
        elif hasattr(action, name):
            val = getattr(action, name)
            if val is not None:
                return val
    return default


# ‑‑‑ Redline XML Formatter ‑‑‑
class RedlineXMLFormatter:
    """Applies xmldiff actions using standardized diff:* attributes."""

    def __init__(self, **kwargs):
        self.detect_moves = kwargs.get("detect_moves", True)
        self.normalize = kwargs.get("normalize", formatting.WS_BOTH)
        self._orig_root: Optional[_Element] = None
        self._mod_root: Optional[_Element] = None
        self._annotated_copy_root: Optional[_Element] = None
        self._annotated_copy_tree: Optional[_ElementTree] = None
        self._actions: List[Any] = []
        self._node_map_orig_to_copy: Dict[_Element, _Element] = {}
        self._xpath_cache_orig: Dict[str, List[_Element]] = {}
        self._xpath_cache_mod: Dict[str, List[_Element]] = {}
        self.processed_actions: Dict[str, int] = {
            "insertions": 0,
            "deletions": 0,
            "moves": 0,
            "text_updates": 0,
            "attr_updates": 0,
            "renames": 0,
            "other_changes": 0,
            "errors": 0,
            "inline_insertions": 0,
            "inline_deletions": 0,
        }
        self._attr_changes: Dict[_Element, List[Dict[str, str]]] = {}

    def _reset_state(self):
        self._orig_root = None
        self._mod_root = None
        self._annotated_copy_root = None
        self._annotated_copy_tree = None
        self._actions = []
        self._node_map_orig_to_copy = {}
        self._xpath_cache_orig.clear()
        self._xpath_cache_mod.clear()
        self.processed_actions = {k: 0 for k in self.processed_actions}
        self._attr_changes.clear()

    @staticmethod
    def _add_diff_attribute(elem: _Element, name: str, value: Optional[str] = "true"):
        """Adds a diff:* attribute."""
        if elem is None:
            return
        if not isinstance(elem, _Element):
            return
        qname = f"{{{_DIFF_NS}}}{name}"
        val_str = str(value) if value is not None else ""
        try:
            elem.set(qname, val_str)
        except ValueError as e:
            logger.error(f"Failed to set attr '{qname}'='{val_str}' on <{elem.tag}>: {e}")

    def _add_attribute_change_detail(self, node: _Element, change_info: Dict[str, str]):
        if node not in self._attr_changes:
            self._attr_changes[node] = []
        self._attr_changes[node].append(change_info)

    def _aggregate_attribute_changes(self):
        for node, changes in self._attr_changes.items():
            if node is None or not changes:
                continue
            try:
                change_summary = json.dumps(changes)
                self._add_diff_attribute(node, "attributes", change_summary)
            except (TypeError, ValueError) as e:
                logger.error(f"Could not serialize attr changes for {node.tag}: {e}")
                self._add_diff_attribute(node, "attributes", "[Serialization Error]")

    def _get_node_from_xpath(self, xpath: str, tree_type: str) -> Optional[_Element]:
        if not xpath:
            return None
        root = (
            self._orig_root
            if tree_type == "original"
            else self._mod_root
            if tree_type == "modified"
            else None
        )
        cache = (
            self._xpath_cache_orig
            if tree_type == "original"
            else self._xpath_cache_mod
            if tree_type == "modified"
            else None
        )
        if root is None or cache is None:
            return None
        if xpath in cache:
            nodes = cache[xpath]
            return nodes[0] if nodes else None
        try:
            adjusted_xpath = xpath[2:] if xpath.startswith("/0/") else xpath
            nodes = root.xpath(adjusted_xpath)
            element_nodes = [n for n in nodes if isinstance(n, _Element)]
            cache[xpath] = element_nodes
            return element_nodes[0] if element_nodes else None
        except Exception:
            cache[xpath] = []
            return None

    def _get_corresponding_node_in_copy(self, orig_node: _Element) -> Optional[_Element]:
        if orig_node is None:
            return None
        if orig_node in self._node_map_orig_to_copy:
            return self._node_map_orig_to_copy[orig_node]
        if self._orig_root is not None and self._annotated_copy_root is not None:
            try:
                orig_xpath = self._orig_root.getroottree().getpath(orig_node)
                if orig_xpath:
                    copy_nodes = self._annotated_copy_root.xpath(orig_xpath)
                    if copy_nodes and isinstance(copy_nodes[0], _Element):
                        self._node_map_orig_to_copy[orig_node] = copy_nodes[0]
                        return copy_nodes[0]
            except Exception:
                pass
        return None

    def _build_initial_node_map(self):
        if self._orig_root is None or self._annotated_copy_root is None:
            return
        self._node_map_orig_to_copy.clear()
        orig_iter = self._orig_root.iter()
        copy_iter = self._annotated_copy_root.iter()
        try:
            while True:
                orig_node = next(orig_iter)
                copy_node = next(copy_iter)
                if isinstance(orig_node, _Element) and isinstance(copy_node, _Element):
                    if (
                        hasattr(orig_node, "tag")
                        and hasattr(copy_node, "tag")
                        and orig_node.tag == copy_node.tag
                    ):
                        self._node_map_orig_to_copy[orig_node] = copy_node
        except StopIteration:
            pass
        except Exception as e:
            logger.error(f"Error during initial node mapping: {e}")
        logger.debug(f"Built initial node map with {len(self._node_map_orig_to_copy)} entries.")

    def _find_node_in_copy_by_xpath(self, xpath: str) -> Optional[_Element]:
        if not xpath or self._annotated_copy_root is None:
            return None
        try:
            adjusted_xpath = xpath[2:] if xpath.startswith("/0/") else xpath
            nodes = self._annotated_copy_root.xpath(adjusted_xpath)
            if nodes and isinstance(nodes[0], _Element):
                return nodes[0]
            elif nodes:
                try:
                    parent = nodes[0].getparent()
                    if isinstance(parent, _Element):
                        return parent
                except AttributeError:
                    pass
                return None
            else:
                return None
        except Exception:
            return None

    # --- Action Handlers ---

    def _handle_delete_node(self, action: DeleteNode):
        node_xpath = _safe_get_attr(action, "node", "node_xpath", "target")
        if not node_xpath:
            logger.error(f"DeleteNode missing XPath: {action}")
            self.processed_actions["errors"] += 1
            return
        orig_node = self._get_node_from_xpath(node_xpath, "original")
        if orig_node is None:
            logger.warning(f"DeleteNode: Original node {node_xpath} not found.")
            return
        copy_node = self._get_corresponding_node_in_copy(orig_node)
        if copy_node is None:
            logger.warning(f"DeleteNode: Copy node for {node_xpath} not found.")
            return

        move_id = _safe_get_attr(action, "move_id")
        if not move_id:
            move_node = next(
                (
                    a
                    for a in self._actions
                    if isinstance(a, MoveNode) and _safe_get_attr(a, "node", "source") == node_xpath
                ),
                None,
            )
            move_id = _safe_get_attr(move_node, "move_id")

        if move_id:
            self._add_diff_attribute(copy_node, "op", "move-source")
            self._add_diff_attribute(copy_node, "move-id", move_id)
        else:
            self._add_diff_attribute(copy_node, "op", "delete")
            self.processed_actions["deletions"] += 1

    def _handle_insert_node(self, action: InsertNode):
        parent_xpath = _safe_get_attr(action, "parent_xpath", "target")
        node_structure = _safe_get_attr(action, "node")
        tag = _safe_get_attr(action, "tag")
        pos = _safe_get_attr(action, "pos", "position")
        sibling_xpath = _safe_get_attr(action, "sibling_xpath")
        if not parent_xpath:
            logger.error(f"InsertNode missing parent: {action}")
            self.processed_actions["errors"] += 1
            return

        node_to_insert = None
        if node_structure is not None and isinstance(node_structure, _Element):
            try:
                # Attempt to clone from the action object first
                node_to_insert = deepcopy(node_structure)  # Use deepcopy
                # node_to_insert = etree.fromstring(etree.tostring(node_structure))
            except Exception as e:
                logger.error(f"InsertNode clone failed: {e}")

        # # ➜ REMOVE THIS BLOCK ------------------------------------------
        # if node_to_insert is None:
        #     # ➜ NEW: find the inserted node in the MODIFIED tree ----------------
        #     if parent_xpath:
        #         mod_parent = self._get_node_from_xpath(parent_xpath, "modified")
        #         if mod_parent is not None:
        #             try:
        #                 idx = int(pos) if str(pos).isdigit() else len(mod_parent) - 1
        #                 # Make sure to clone the node from the modified tree
        #                 potential_node = mod_parent[idx]
        #                 if potential_node is not None:
        #                      node_to_insert = deepcopy(potential_node)
        #                      logger.debug(f"InsertNode: Fetched node <{node_to_insert.tag}> from modified tree.")
        #                 else:
        #                      node_to_insert = None
        #             except Exception as e:
        #                 logger.warning(f"InsertNode: Failed to fetch node from modified tree at {parent_xpath}[{pos}]: {e}")
        #                 node_to_insert = None
        # # ---------------------------------------------------------------

        # If cloning/fetching failed, create a placeholder
        if node_to_insert is None:
            if tag:
                attrs = _safe_get_attr(action, "attrib", "attributes", default={}) or {}
                node_to_insert = etree.Element(tag, attrs)
                # Make placeholder text more distinct
                node_to_insert.text = f"[Placeholder: Inserted <{tag}> content missing]"
                logger.warning(
                    f"InsertNode created placeholder <{tag}> because node structure was missing in action and couldn't be fetched."
                )
            else:
                logger.error("InsertNode failed: No structure/tag provided in action.")
                self.processed_actions["errors"] += 1
                return

        move_id = _safe_get_attr(action, "move_id")
        is_move_target = bool(move_id)
        if not move_id:
            move_node = next(
                (
                    a
                    for a in self._actions
                    if isinstance(a, MoveNode)
                    and _safe_get_attr(a, "target") == parent_xpath
                    and str(_safe_get_attr(a, "pos", "position")) == str(pos)
                ),
                None,
            )
            if move_node:
                move_id = _safe_get_attr(move_node, "move_id")
                is_move_target = bool(move_id)
                if not move_id:
                    logger.warning(f"Insert seems move target but MoveNode lacks ID: {action}")

        if is_move_target and move_id:
            if not any(isinstance(a, MoveNode) for a in self._actions):
                self.processed_actions["moves"] += 1
            pass
        else:
            self._add_diff_attribute(node_to_insert, "op", "insert")
            if is_move_target:
                self.processed_actions["errors"] += 1
                logger.warning(
                    f"Marking node <{node_to_insert.tag}> as insert (was move target w/o ID)."
                )
            self.processed_actions["insertions"] += 1

        if not (is_move_target and any(isinstance(a, MoveNode) for a in self._actions)):
            target_node_in_copy = self._find_node_in_copy_by_xpath(
                parent_xpath if pos == "into" or isinstance(pos, int) else sibling_xpath
            )
            if target_node_in_copy is None:
                logger.error(
                    f"InsertNode: Target node not found in COPY. XPath: '{parent_xpath if pos == 'into' or isinstance(pos, int) else sibling_xpath}'."
                )
                self.processed_actions["errors"] += 1
                return
            try:
                if pos == "into" or isinstance(pos, int):
                    parent = target_node_in_copy
                    idx = (
                        int(pos)
                        if isinstance(pos, int) or (isinstance(pos, str) and pos.isdigit())
                        else len(parent)
                    )
                    idx = max(0, min(idx, len(parent)))
                    parent.insert(idx, node_to_insert)
                elif pos == "before":
                    sibling = target_node_in_copy
                    parent = sibling.getparent()
                    parent.insert(parent.index(sibling), node_to_insert)
                elif pos == "after":
                    sibling = target_node_in_copy
                    parent = sibling.getparent()
                    parent.insert(parent.index(sibling) + 1, node_to_insert)
                else:
                    raise ValueError(f"Unknown pos '{pos}'")
            except Exception as e:
                logger.exception(f"InsertNode insert error: {e}")
                self.processed_actions["errors"] += 1

    def _handle_move_node(self, action: MoveNode):
        """Handle move: Ensure source marked, insert clone at target."""
        src_xpath = _safe_get_attr(action, "node", "source")
        tgt_xpath = _safe_get_attr(action, "target")
        pos = _safe_get_attr(
            action, "pos", "position", default="into"
        )  # Keep default 'into' if missing
        move_id = _safe_get_attr(action, "move_id")

        if not src_xpath or not tgt_xpath or not move_id:
            # Use ToolError for critical diff engine issues
            raise ToolError(
                f"xmldiff produced a MoveNode without complete data "
                f"(src={src_xpath!r}, tgt={tgt_xpath!r}, id={move_id!r}). "
                "This indicates malfunction in the diff stage.",
                code="DIFF_ENGINE_ERROR",
            )

        orig_src_node = self._get_node_from_xpath(src_xpath, "original")
        if orig_src_node is None:
            logger.error(f"MoveNode {move_id}: Original source node {src_xpath} not found.")
            self.processed_actions["errors"] += 1
            return
        copy_src_node = self._get_corresponding_node_in_copy(orig_src_node)
        if copy_src_node is None:
            # Log error but attempt to continue if possible - maybe source was deleted then moved? Unlikely but cover edge case.
            logger.error(
                f"MoveNode {move_id}: Corresponding copy source node for {src_xpath} not found."
            )
            # If the source isn't in the copy, we can't mark it, but we still need to insert the target.
            # No need to return here, proceed to insert the target.
            # self.processed_actions["errors"] += 1 # Maybe not an error if source was already removed by another action?
        else:
            # Ensure the source node in the copy is marked correctly
            # It might have been marked by _handle_delete_node already if xmldiff emits Delete then Move
            # Check if marking is already correct to avoid redundant logging/work
            if (
                copy_src_node.get(f"{{{_DIFF_NS}}}op") != "move-source"
                or copy_src_node.get(f"{{{_DIFF_NS}}}move-id") != move_id
            ):
                logger.debug(f"MoveNode {move_id}: Marking source node {src_xpath} in copy.")
                self._add_diff_attribute(copy_src_node, "op", "move-source")
                self._add_diff_attribute(copy_src_node, "move-id", move_id)

        # --- Determine the node to clone ---
        # The goal is to clone the node *as it exists in the modified document*
        # The `action` tells us where it ended up (tgt_xpath, pos).
        node_to_clone = None
        mod_target_parent = self._get_node_from_xpath(tgt_xpath, "modified")

        if mod_target_parent is not None and isinstance(pos, int) and pos >= 0:
            try:
                # Get the actual node from the modified tree at the target position
                node_to_clone = mod_target_parent[pos]
                logger.debug(
                    f"MoveNode {move_id}: Found node to clone in MODIFIED tree at {tgt_xpath}[{pos}]."
                )
            except IndexError:
                logger.warning(
                    f"MoveNode {move_id}: Index {pos} out of bounds for target parent {tgt_xpath} in MODIFIED tree. Parent has {len(mod_target_parent)} children."
                )
            except Exception as e:
                logger.warning(
                    f"MoveNode {move_id}: Error accessing node at {tgt_xpath}[{pos}] in MODIFIED tree: {e}"
                )
        elif mod_target_parent is not None and pos == "into":  # Handle insertion 'into' as append
            try:
                # If pos is 'into', it usually implies appending. The moved node would be the last child.
                # However, xmldiff usually gives an integer position for moves.
                # Let's try finding based on the source node's ID if possible, as a fallback.
                mod_node_with_same_id = mod_target_parent.xpath(
                    f".//*[@data-diff-id='{orig_src_node.get('data-diff-id')}']"
                )
                if mod_node_with_same_id:
                    node_to_clone = mod_node_with_same_id[0]
                    logger.debug(
                        f"MoveNode {move_id}: Found node to clone in MODIFIED tree based on ID matching source ID within {tgt_xpath}."
                    )
                else:
                    logger.warning(
                        f"MoveNode {move_id}: Position is '{pos}', couldn't find node to clone in MODIFIED target parent {tgt_xpath} by index or ID."
                    )

            except Exception as e:
                logger.warning(
                    f"MoveNode {move_id}: Error finding node in MODIFIED target parent {tgt_xpath} for pos='{pos}': {e}"
                )

        if node_to_clone is None:
            # Fallback: Clone the original source node. This might lose internal changes.
            node_to_clone = orig_src_node
            logger.warning(
                f"MoveNode {move_id}: Could not find moved node in MODIFIED tree at {tgt_xpath}:{pos}. "
                f"Falling back to cloning ORIGINAL source node {src_xpath}. Internal changes might be lost."
            )

        # --- Clone and prepare the node for insertion ---
        try:
            # Use deepcopy which might be more robust for lxml elements than fromstring(tostring)
            cloned_node_for_insert = deepcopy(node_to_clone)
            if cloned_node_for_insert is None:
                raise ValueError("Deepcopy resulted in None")  # noqa: E701
        except Exception as e:
            logger.error(f"MoveNode {move_id}: Cloning node failed: {e}")
            self.processed_actions["errors"] += 1
            return

        # --- Clean and mark the cloned node ---
        # Remove any pre-existing diff attributes from the clone and its descendants
        for el in cloned_node_for_insert.xpath(".//* | ."):  # Iterate over self and descendants
            if isinstance(el, _Element):
                for name in list(el.attrib):
                    if name.startswith(f"{{{_DIFF_NS}}}"):
                        del el.attrib[name]
                    # Also remove the synthetic ID from the clone to avoid collisions if diff runs again
                    if name == "data-diff-id":
                        del el.attrib[name]

        # Mark the root of the clone as the move target
        self._add_diff_attribute(cloned_node_for_insert, "op", "move-target")
        self._add_diff_attribute(cloned_node_for_insert, "move-id", move_id)

        # --- Insert the cloned node into the copy tree ---
        target_node_in_copy = self._find_node_in_copy_by_xpath(tgt_xpath)
        if target_node_in_copy is None:
            logger.error(
                f"MoveNode {move_id}: Target parent node {tgt_xpath} not found in COPY tree for insertion."
            )
            self.processed_actions["errors"] += 1
            # Attempt to insert into the root as a last resort? Or just fail? Let's fail.
            return

        try:
            if isinstance(pos, int) and pos >= 0:
                # Insert at the specific index within the target parent found in the copy tree
                parent = target_node_in_copy
                # Clamp index to valid range for insertion
                idx = max(0, min(int(pos), len(parent)))
                parent.insert(idx, cloned_node_for_insert)
                logger.debug(
                    f"MoveNode {move_id}: Inserted move-target clone into copy tree at {tgt_xpath}[{idx}]."
                )
                self.processed_actions["moves"] += 1
            elif pos == "into":  # Handle 'into' - append to the target node
                parent = target_node_in_copy
                parent.append(cloned_node_for_insert)
                logger.debug(
                    f"MoveNode {move_id}: Appended move-target clone into copy tree node {tgt_xpath}."
                )
                self.processed_actions["moves"] += 1
            else:
                # This case (e.g., pos='before'/'after') shouldn't happen with MoveNode from xmldiff typically,
                # as it uses parent path + index. Log an error if it does.
                logger.error(
                    f"MoveNode {move_id}: Unsupported position '{pos}' for insertion. Expected integer or 'into'."
                )
                self.processed_actions["errors"] += 1
        except Exception as e:
            logger.exception(
                f"MoveNode {move_id}: Insertion of cloned node into copy tree failed: {e}"
            )
            self.processed_actions["errors"] += 1

    def _handle_update_text(self, action: Union[UpdateTextIn, UpdateTextBefore, UpdateTextAfter]):
        xpath = _safe_get_attr(action, "node", "node_xpath")
        new_text = _safe_get_attr(action, "text", "new", "new_text", default="")
        if not xpath:
            logger.error(f"{type(action).__name__} missing XPath: {action}")
            self.processed_actions["errors"] += 1
            return
        if _fine_grained_text_actions and type(action) in (_INSERT_TEXT_CLS + _DELETE_TEXT_CLS):
            return

        copy_node = self._find_node_in_copy_by_xpath(xpath)
        if copy_node is None:
            logger.warning(f"{type(action).__name__}: Node {xpath} not found in COPY.")
            return

        orig_node = self._get_node_from_xpath(xpath, "original")
        actual_old_text = "[Unknown Old Text]"
        update_type = "text"
        if orig_node is not None:
            if isinstance(action, UpdateTextIn):
                actual_old_text = orig_node.text or ""
                update_type = "text"
            elif UpdateTextAfter is not None and isinstance(action, UpdateTextAfter):
                actual_old_text = orig_node.tail or ""
                update_type = "tail"
            elif UpdateTextBefore is not None and isinstance(action, UpdateTextBefore):
                actual_old_text = _safe_get_attr(action, "old", "old_text", default="[?]")
                update_type = "before"
            else:
                actual_old_text = orig_node.text or ""
        else:
            actual_old_text = _safe_get_attr(
                action, "old", "old_text", default="[Missing Orig Node]"
            )

        norm_old = " ".join(str(actual_old_text).split())
        norm_new = " ".join(str(new_text).split())
        if norm_old == norm_new:
            return

        if update_type == "text":
            copy_node.text = new_text
            if len(copy_node) > 0:
                for child in list(copy_node):
                    copy_node.remove(child)
            self._add_diff_attribute(copy_node, "op", "update-text")
            self._add_diff_attribute(copy_node, "old-value", actual_old_text)
            self.processed_actions["text_updates"] += 1
        elif update_type == "tail":
            copy_node.tail = new_text
            self._add_diff_attribute(copy_node, "op", "update-tail")
            self._add_diff_attribute(copy_node, "old-value", actual_old_text)
            self.processed_actions["text_updates"] += 1
        elif update_type == "before":
            self._add_diff_attribute(copy_node, "op", "update-text-context")
            self._add_diff_attribute(copy_node, "detail", "before")
            self._add_diff_attribute(copy_node, "new-value", new_text)
            self._add_diff_attribute(copy_node, "old-value", actual_old_text)
            self.processed_actions["text_updates"] += 1
            logger.warning(f"UpdateTextBefore marked on node {xpath}.")

    def _handle_attr_change(
        self, action: Union[UpdateAttrib, InsertAttrib, DeleteAttrib, RenameAttrib]
    ):
        xpath = _safe_get_attr(action, "node", "node_xpath")
        if not xpath:
            logger.error(f"{type(action).__name__} missing XPath: {action}")
            self.processed_actions["errors"] += 1
            return
        copy_node = self._find_node_in_copy_by_xpath(xpath)
        if copy_node is None:
            logger.warning(f"{type(action).__name__}: Node {xpath} not found in COPY.")
            return
        orig_node = self._get_node_from_xpath(xpath, "original")

        change_info = {}
        processed = False
        try:
            if isinstance(action, UpdateAttrib):
                name = _safe_get_attr(action, "name")
                new_val = _safe_get_attr(action, "value", "new", default="")
                if name is None:
                    logger.error(f"UpdateAttrib missing name: {action}")
                    return
                old_val = orig_node.get(name) if orig_node is not None else "[?]"
                if old_val != new_val:
                    copy_node.set(name, new_val)
                    change_info = {"op": "update", "name": name, "old": old_val, "new": new_val}
                    self._add_attribute_change_detail(copy_node, change_info)
                    self.processed_actions["attr_updates"] += 1
                    processed = True
            elif isinstance(action, InsertAttrib):
                name = _safe_get_attr(action, "name")
                value = _safe_get_attr(action, "value", default="")
                if name is None:
                    logger.error(f"InsertAttrib missing name: {action}")
                    return
                copy_node.set(name, value)
                change_info = {"op": "insert", "name": name, "new": value}
                self._add_attribute_change_detail(copy_node, change_info)
                self.processed_actions["attr_updates"] += 1
                processed = True
            elif isinstance(action, DeleteAttrib):
                name = _safe_get_attr(action, "name")
                if name is None:
                    logger.error(f"DeleteAttrib missing name: {action}")
                    return
                old_val = orig_node.get(name) if orig_node is not None else "[?]"
                if name in copy_node.attrib:
                    del copy_node.attrib[name]
                change_info = {"op": "delete", "name": name, "old": old_val}
                self._add_attribute_change_detail(copy_node, change_info)
                self.processed_actions["attr_updates"] += 1
                processed = True
            elif isinstance(action, RenameAttrib):
                old_n = _safe_get_attr(action, "old_name")
                new_n = _safe_get_attr(action, "new_name")
                if not old_n or not new_n:
                    logger.error(f"RenameAttrib missing names: {action}")
                    return
                value = orig_node.get(old_n) if orig_node is not None else "[?]"
                if old_n in copy_node.attrib:
                    del copy_node.attrib[old_n]
                copy_node.set(new_n, value)
                change_info = {"op": "rename", "old_name": old_n, "new_name": new_n, "value": value}
                self._add_attribute_change_detail(copy_node, change_info)
                self.processed_actions["attr_updates"] += 1
                self.processed_actions["renames"] += 1
                processed = True
            if processed and copy_node.get(f"{{{_DIFF_NS}}}op") is None:
                self._add_diff_attribute(copy_node, "op", "update-attrib")
        except Exception as e:
            logger.exception(f"Attr change error for {xpath}: {e}")
            self.processed_actions["errors"] += 1

    def _handle_rename_node(self, action: RenameNode):
        xpath = _safe_get_attr(action, "node", "node_xpath")
        new_tag = _safe_get_attr(action, "new_tag", "new_name")
        if not xpath or not new_tag:
            logger.error(f"RenameNode missing xpath/new_tag: {action}")
            self.processed_actions["errors"] += 1
            return
        copy_node = self._find_node_in_copy_by_xpath(xpath)
        if copy_node is None:
            logger.warning(f"RenameNode: Node {xpath} not found in COPY.")
            return
        orig_node = self._get_node_from_xpath(xpath, "original")
        old_tag = orig_node.tag if orig_node is not None else copy_node.tag
        if old_tag != new_tag:
            copy_node.tag = new_tag
            self._add_diff_attribute(copy_node, "op", "rename-node")
            self._add_diff_attribute(copy_node, "old-value", old_tag)
            self._add_diff_attribute(copy_node, "new-value", new_tag)
            self.processed_actions["renames"] += 1

    def _handle_insert_text_node(
        self, action: Union[InsertTextIn, InsertTextBefore, InsertTextAfter]
    ):
        xpath = _safe_get_attr(action, "node", "node_xpath")
        text = _safe_get_attr(action, "text", "value", default="")
        if not xpath:
            logger.error(f"{type(action).__name__} missing xpath: {action}")
            self.processed_actions["errors"] += 1
            return
        copy_node = self._find_node_in_copy_by_xpath(xpath)
        if copy_node is None:
            logger.warning(f"{type(action).__name__}: Ref node {xpath} missing in copy.")
            return
        ins_el = etree.Element("ins", attrib={"class": "diff-insert-text"})
        ins_el.text = text
        try:
            if InsertTextBefore is not None and isinstance(action, InsertTextBefore):
                parent = copy_node.getparent()
                parent.insert(parent.index(copy_node), ins_el)
                self.processed_actions["inline_insertions"] += 1
            elif InsertTextAfter is not None and isinstance(action, InsertTextAfter):
                ins_el.tail = copy_node.tail
                copy_node.tail = None
                copy_node.addnext(ins_el)
                self.processed_actions["inline_insertions"] += 1
            elif InsertTextIn is not None and isinstance(action, InsertTextIn):
                pos = _safe_get_attr(action, "pos", default=len(copy_node))
                idx = int(pos) if isinstance(pos, int) or str(pos).isdigit() else len(copy_node)
                idx = max(0, min(idx, len(copy_node)))
                copy_node.insert(idx, ins_el)
                self.processed_actions["inline_insertions"] += 1
            else:
                logger.warning(f"Unhandled InsertText: {type(action).__name__}.")
                copy_node.append(ins_el)
                self.processed_actions["inline_insertions"] += 1
        except Exception as e:
            logger.exception(f"InsertText error: {e}")
            self.processed_actions["errors"] += 1

    def _handle_delete_text_node(
        self, action: Union[DeleteTextIn, DeleteTextBefore, DeleteTextAfter]
    ):
        xpath = _safe_get_attr(action, "node", "node_xpath")
        text = _safe_get_attr(action, "text", "value")
        if not xpath or text is None:
            logger.error(f"{type(action).__name__} missing xpath/text: {action}")
            self.processed_actions["errors"] += 1
            return
        copy_node = self._find_node_in_copy_by_xpath(xpath)
        if copy_node is None:
            logger.warning(f"{type(action).__name__}: Ref node {xpath} missing in copy.")
            return
        del_el = etree.Element("del", attrib={"class": "diff-delete-text"})
        del_el.text = text
        try:
            if DeleteTextBefore is not None and isinstance(action, DeleteTextBefore):
                parent = copy_node.getparent()
                parent.insert(parent.index(copy_node), del_el)
                self.processed_actions["inline_deletions"] += 1
            elif DeleteTextAfter is not None and isinstance(action, DeleteTextAfter):
                orig_tail = copy_node.tail
                copy_node.tail = None
                del_el.tail = orig_tail
                copy_node.addnext(del_el)
                self.processed_actions["inline_deletions"] += 1
            elif DeleteTextIn is not None and isinstance(action, DeleteTextIn):
                pos = _safe_get_attr(action, "pos", default=0)
                idx = int(pos) if isinstance(pos, int) or str(pos).isdigit() else 0
                idx = max(0, min(idx, len(copy_node)))
                if copy_node.text and text in copy_node.text:
                    copy_node.text = copy_node.text.replace(text, "", 1)
                copy_node.insert(idx, del_el)
                self.processed_actions["inline_deletions"] += 1
            else:
                logger.warning(f"Unhandled DeleteText: {type(action).__name__}.")
                copy_node.insert(0, del_el)
                self.processed_actions["inline_deletions"] += 1
        except Exception as e:
            logger.exception(f"DeleteText error: {e}")
            self.processed_actions["errors"] += 1

    # --- Main Formatting Method ---

    def format(
        self, actions: List[Any], orig_doc: _ElementTree, mod_doc: _ElementTree
    ) -> _ElementTree:
        """Applies diff actions to a copy of orig_doc."""
        self._reset_state()
        self._actions = actions
        logger.debug(f"Formatter init with {len(actions)} actions.")
        self._orig_root = orig_doc.getroot()
        self._mod_root = mod_doc.getroot()
        if self._orig_root is None or self._mod_root is None:
            raise ValueError("Docs missing root.")
        try:
            self._annotated_copy_tree = deepcopy(orig_doc)
            self._annotated_copy_root = self._annotated_copy_tree.getroot()
            assert self._annotated_copy_root is not None
        except Exception as e:
            logger.exception("Deepcopy failed.")
            raise RuntimeError("Copy failed.") from e
        self._build_initial_node_map()

        etree.register_namespace(_DIFF_PREFIX, _DIFF_NS)
        if (
            self._annotated_copy_root is not None
            and _DIFF_PREFIX not in self._annotated_copy_root.nsmap
        ):
            new_nsmap = self._annotated_copy_root.nsmap.copy()
            new_nsmap[_DIFF_PREFIX] = _DIFF_NS
            new_root = etree.Element(
                self._annotated_copy_root.tag,
                nsmap=new_nsmap,
                attrib=self._annotated_copy_root.attrib,
            )
            new_root.text = self._annotated_copy_root.text
            new_root.tail = self._annotated_copy_root.tail
            for child in self._annotated_copy_root:
                new_root.append(child)
            self._annotated_copy_tree._setroot(new_root)
            self._annotated_copy_root = new_root
            logger.debug(f"Registered '{_DIFF_PREFIX}' ns.")

        action_handlers = {
            DeleteNode: self._handle_delete_node,
            InsertNode: self._handle_insert_node,
            MoveNode: self._handle_move_node,
            UpdateTextIn: self._handle_update_text,
            UpdateAttrib: self._handle_attr_change,
            InsertAttrib: self._handle_attr_change,
            DeleteAttrib: self._handle_attr_change,
            RenameAttrib: self._handle_attr_change,
        }
        if RenameNode:
            action_handlers[RenameNode] = self._handle_rename_node
        if UpdateTextBefore:
            action_handlers[UpdateTextBefore] = self._handle_update_text
        if UpdateTextAfter:
            action_handlers[UpdateTextAfter] = self._handle_update_text
        if _fine_grained_text_actions:
            for cls in _INSERT_TEXT_CLS:
                action_handlers[cls] = self._handle_insert_text_node
            for cls in _DELETE_TEXT_CLS:
                action_handlers[cls] = self._handle_delete_text_node

        logger.info(f"Applying {len(actions)} actions to the document copy...")
        for i, action in enumerate(actions):
            atype = type(action)
            handler = action_handlers.get(atype)
            if handler:
                try:
                    handler(action)
                except Exception:
                    logger.exception(f"Handler error #{i + 1} ({atype.__name__}): {action}")
                    self.processed_actions["errors"] += 1
            elif atype == InsertComment:
                pass
            else:
                logger.warning(f"Unhandled action: {atype.__name__}")
                self.processed_actions["other_changes"] += 1

        self._aggregate_attribute_changes()

        total = sum(
            v for k, v in self.processed_actions.items() if k not in ["total_changes", "errors"]
        )
        self.processed_actions["total_changes"] = total
        logger.info(f"Action processing complete. Stats: {self.processed_actions}")
        if self._annotated_copy_tree is None:
            raise RuntimeError("Formatting failed, tree is None.")
        return self._annotated_copy_tree


# ─────────────────────────────────────────────────────────────────────────────
#                     Markdown summary-generation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _node_plain_text(node: Optional[_Element], *, max_len: int = 120) -> str:
    if node is None:
        return "[Node is None]"
    try:
        texts = [t for t in node.xpath(".//text()[not(parent::script) and not(parent::style)]")]
        txt = " ".join(t.strip() for t in texts if t.strip())
        txt = re.sub(r"\s+", " ", txt).strip()
    except Exception as e:
        logger.warning(f"Text extract error: {e}")
        txt = " ".join(node.itertext()).strip()
        txt = re.sub(r"\s+", " ", txt).strip()
    return textwrap.shorten(txt, max_len, placeholder="…") if max_len else txt


def _get_element_by_xpath_from_tree(xpath: str, tree: _ElementTree) -> Optional[_Element]:
    if not xpath or tree is None:
        return None
    root = tree.getroot()
    if root is None:
        return None
    try:
        nodes = root.xpath(xpath[2:] if xpath.startswith("/0/") else xpath)
        return next((n for n in nodes if isinstance(n, _Element)), None)
    except Exception:
        return None


def _generate_markdown_summary(
    *,
    orig_doc: _ElementTree,
    mod_doc: _ElementTree,
    actions: List[Any],
    context_chars: int = 120,
) -> str:
    ts = _dt.datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"
    lines: List[str] = [f"# Detected Redline Differences ({ts})\n"]

    if orig_doc is None or mod_doc is None:
        return "# Error: Missing docs."

    processed_move_ids: set[str] = set()
    move_actions: Dict[str, Dict[str, Any]] = {}

    # --------------------------------------------------------------------- #
    # Pass 1 – collect information about moves
    # --------------------------------------------------------------------- #
    for a in actions:
        if isinstance(a, MoveNode):
            mid = _safe_get_attr(a, "move_id")
            src = _safe_get_attr(a, "node", "source")
            tgt = _safe_get_attr(a, "target")
            pos = _safe_get_attr(a, "pos", "position")
            if mid and src and tgt:
                move_actions.setdefault(mid, {})["src"] = src
                move_actions[mid].update({"tgt": tgt, "pos": pos, "found": True})
                processed_move_ids.add(mid)

        elif isinstance(a, DeleteNode):
            mid = _safe_get_attr(a, "move_id")
            src = _safe_get_attr(a, "node", "node_xpath", "target")
            if mid and src and mid not in processed_move_ids:
                move_actions.setdefault(mid, {})["src"] = src
                move_actions[mid].update({"tgt": "?", "pos": "?", "found": False})
                processed_move_ids.add(mid)

        elif isinstance(a, InsertNode):
            mid = _safe_get_attr(a, "move_id")
            tgt = _safe_get_attr(a, "parent_xpath", "target")
            pos = _safe_get_attr(a, "pos", "position")
            if mid and tgt and mid in move_actions and not move_actions[mid]["found"]:
                move_actions[mid].update({"tgt": tgt, "pos": pos})

    # --------------------------------------------------------------------- #
    # Moves section
    # --------------------------------------------------------------------- #
    if move_actions:
        lines.append("## Moves\n")
        for mid, info in move_actions.items():
            src_elem = _get_element_by_xpath_from_tree(info["src"], orig_doc)
            tgt_elem = _get_element_by_xpath_from_tree(info["tgt"], mod_doc)
            loc = f"into <{tgt_elem.tag}>" if tgt_elem is not None else f"near {info['tgt']}"
            content_txt = _node_plain_text(src_elem, max_len=context_chars) if src_elem else "[?]"

            lines.extend(
                [
                    f"### Move ID: `{mid}`",
                    f"- **From:** `{info['src']}`",
                    f"- **To:** `{loc}` (Pos: {info['pos']})",
                    "- **Content:**",
                    "  ```text",
                    f"  {content_txt}",
                    "  ```\n",
                ]
            )
        lines.append("---\n")

    # --------------------------------------------------------------------- #
    # Headings map
    # --------------------------------------------------------------------- #
    hdrs = {
        InsertNode: "## Insertions\n",
        DeleteNode: "## Deletions\n",
        UpdateTextIn: "## Text Updates\n",
        UpdateAttrib: "## Attr Updates\n",
        InsertAttrib: "## Attr Updates\n",
        DeleteAttrib: "## Attr Updates\n",
        RenameAttrib: "## Attr Updates\n",
        RenameNode: "## Node Renames\n",
    }

    cur_sec = None

    # --------------------------------------------------------------------- #
    # Main pass – every non-move action
    # --------------------------------------------------------------------- #
    for a in actions:
        atype = type(a)
        mid = _safe_get_attr(a, "move_id")
        if isinstance(a, MoveNode) or (mid and mid in processed_move_ids):
            continue

        # heading management
        if atype in hdrs:
            if hdrs[atype] != cur_sec:
                if cur_sec:
                    lines.append("---\n")
                lines.append(hdrs[atype])
                cur_sec = hdrs[atype]
        else:
            if cur_sec != "## Other Changes\n":
                if cur_sec:
                    lines.append("---\n")
                lines.append("## Other Changes\n")
                cur_sec = "## Other Changes\n"

        try:
            summary: List[str] = []

            # ------------------------------------------------------------- #
            # INSERT NODE
            # ------------------------------------------------------------- #
            if isinstance(a, InsertNode):
                pxp = _safe_get_attr(a, "parent_xpath", "target")
                pos = _safe_get_attr(a, "pos", "position", default="N/A")
                node_s = _safe_get_attr(a, "node")
                tag = node_s.tag if node_s is not None else _safe_get_attr(a, "tag") or "[?]"
                if node_s is not None:
                    raw = etree.tostring(
                        node_s, pretty_print=False, encoding="unicode", method="html"
                    ).strip()
                    content = textwrap.shorten(raw, context_chars * 2, placeholder="…")
                else:
                    content = "[No structure]"
                summary = [
                    f"### Inserted `<{tag}>`",
                    f"- **Location:** Into `{pxp}` (Pos: {pos})",
                    "- **Content:**",
                    "  ```html",
                    f"  {content}",
                    "  ```\n",
                ]

            # ------------------------------------------------------------- #
            # DELETE NODE
            # ------------------------------------------------------------- #
            elif isinstance(a, DeleteNode):
                xp = _safe_get_attr(a, "node", "node_xpath", "target")
                onode = _get_element_by_xpath_from_tree(xp, orig_doc)
                tag = onode.tag if onode else "[?]"
                content = (
                    _node_plain_text(onode, max_len=context_chars) if onode is not None else "[?]"
                )
                summary = [
                    f"### Deleted `<{tag}>`",
                    f"- **Location:** `{xp}`",
                    "- **Content:**",
                    "  ```text",
                    f"  {content}",
                    "  ```\n",
                ]

            # ------------------------------------------------------------- #
            # TEXT UPDATE
            # ------------------------------------------------------------- #
            elif isinstance(a, UpdateTextIn):
                xp = _safe_get_attr(a, "node", "node_xpath")
                onode = _get_element_by_xpath_from_tree(xp, orig_doc)
                mnode = _get_element_by_xpath_from_tree(xp, mod_doc)
                old = _node_plain_text(onode, max_len=context_chars) if onode is not None else "[?]"
                new = _node_plain_text(mnode, max_len=context_chars) if mnode is not None else "[?]"
                tag = (
                    onode.tag if onode is not None else (mnode.tag if mnode is not None else "[?]")
                )
                if old != new:
                    summary = [
                        f"### Text Change in `<{tag}>`",
                        f"- **Location:** `{xp}`",
                        f"- **Old:** `{old}`",
                        f"- **New:** `{new}`\n",
                    ]

            # ------------------------------------------------------------- #
            # ATTRIBUTE-LEVEL CHANGES
            # ------------------------------------------------------------- #
            elif isinstance(a, (UpdateAttrib, InsertAttrib, DeleteAttrib, RenameAttrib)):
                xp = _safe_get_attr(a, "node", "node_xpath")
                onode = _get_element_by_xpath_from_tree(xp, orig_doc)
                mnode = _get_element_by_xpath_from_tree(xp, mod_doc)
                tag = (
                    onode.tag if onode is not None else (mnode.tag if mnode is not None else "[?]")
                )

                details = ""
                if isinstance(a, UpdateAttrib):
                    name = _safe_get_attr(a, "name")
                    old_v = onode.get(name) if onode is not None and name else "[?]"
                    new_v = _safe_get_attr(a, "value", "new")
                    details = f"- **Update:** `{name}`\n- **Old:** `{old_v}`\n- **New:** `{new_v}`"
                elif isinstance(a, InsertAttrib):
                    name = _safe_get_attr(a, "name")
                    val = _safe_get_attr(a, "value")
                    details = f"- **Insert:** `{name}` = `{val}`"
                elif isinstance(a, DeleteAttrib):
                    name = _safe_get_attr(a, "name")
                    old_v = onode.get(name) if onode is not None and name else "[?]"
                    details = f"- **Delete:** `{name}` (was `{old_v}`)"
                elif isinstance(a, RenameAttrib):
                    old_n = _safe_get_attr(a, "old_name")
                    new_n = _safe_get_attr(a, "new_name")
                    val = onode.get(old_n) if onode is not None and old_n else "[?]"
                    details = f"- **Rename:** `{old_n}` → `{new_n}` (value: `{val}`)"

                if details:
                    summary = [f"### Attribute Change in `<{tag}>` (`{xp}`)", details + "\n"]

            # ------------------------------------------------------------- #
            # RENAME NODE
            # ------------------------------------------------------------- #
            elif "RenameNode" in globals() and isinstance(a, RenameNode):
                xp = _safe_get_attr(a, "node", "node_xpath")
                new_tag = _safe_get_attr(a, "new_tag", "new_name")
                onode = _get_element_by_xpath_from_tree(xp, orig_doc)
                old_tag = onode.tag if onode else "[?]"
                summary = [
                    "### Node Rename",
                    f"- **Location:** `{xp}`",
                    f"- **Old Tag:** `{old_tag}`",
                    f"- **New Tag:** `{new_tag}`\n",
                ]

            # append to global list
            lines.extend(summary)

        except Exception as exc:  # pragma: no-cover
            logger.error("Markdown summary error: %s | %s", a, exc, exc_info=True)
            lines.extend(["\n---\nError: " + type(a).__name__ + "\n---\n"])

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#                         Fuzzy move pairing helpers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _NodeFP:
    xpath: str
    tag: str
    text_norm: str
    tokens: List[str]
    shingles: Set[str]
    simhash: int
    anchor_sig: str
    attrs_sig: str
    length: int
    node: _Element
    parent_xpath: Optional[str] = None
    pos: Optional[int] = None
    action_index: int = -1  # index in actions list


def _tokenize_for_fp(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())


def _make_shingles(tokens: List[str], k: int = 4) -> Set[str]:
    if len(tokens) < k:
        return set(tokens) if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(0, len(tokens) - k + 1)}


def _simhash64(items: Set[str]) -> int:
    # Fast 64-bit simhash over shingles/tokens
    if not items:
        return 0
    acc = [0] * 64
    for it in items:
        h = int(hashlib.blake2b(it.encode("utf-8", "replace"), digest_size=8).hexdigest(), 16)
        for b in range(64):
            acc[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b, v in enumerate(acc):
        if v >= 0:
            out |= 1 << b
    return out


def _simhash_sim(a: int, b: int) -> float:
    if a == b:
        return 1.0
    x = a ^ b
    try:
        d = x.bit_count()
    except AttributeError:  # pragma: no cover - Python < 3.8 fallback
        d = bin(x).count("1")
    return 1.0 - (d / 64.0)


def _ancestor_anchor_signature(el: _Element, depth: int = 4) -> str:
    parts: List[str] = []
    p = el.getparent()
    steps = 0
    while p is not None and steps < depth:
        tid = p.get("id") or ""
        cls = p.get("class") or ""
        name = p.tag.lower() if hasattr(p, "tag") else ""
        parts.append(f"{name}#{tid}.{cls}".strip("."))
        p = p.getparent()
        steps += 1
    parts.reverse()
    return "/".join(parts)


def _attrs_signature(el: _Element) -> str:
    if el is None or not hasattr(el, "attrib"):
        return ""
    kv = []
    for k, v in el.attrib.items():
        k_l = k.lower()
        if k_l in _ANCHOR_ATTRS or k_l.startswith("data-"):
            kv.append(f"{k_l}={_normalize_text(v)}")
    kv.sort()
    return "|".join(kv)


def _build_fp_for_element(el: _Element, xpath: str) -> Optional[_NodeFP]:
    if el is None:
        return None
    txt = _get_normalized_full_text(el)
    if len(txt) < _MOVE_MIN_CHARS:
        return None
    toks = _tokenize_for_fp(txt)
    if len(toks) < _MOVE_MIN_TOKENS:
        return None
    sh = _make_shingles(toks, 4) or set(toks)
    tag = el.tag.lower() if hasattr(el, "tag") else ""
    return _NodeFP(
        xpath=xpath,
        tag=tag,
        text_norm=txt,
        tokens=toks,
        shingles=sh,
        simhash=_simhash64(sh),
        anchor_sig=_ancestor_anchor_signature(el),
        attrs_sig=_attrs_signature(el),
        length=len(toks),
        node=el,
    )


def _resolve_insert_element(
    ins_action: Any, mod_doc: _ElementTree
) -> Tuple[Optional[_Element], Optional[str], Optional[int]]:
    parent_xpath = _safe_get_attr(ins_action, "parent_xpath", "target")
    pos = _safe_get_attr(ins_action, "pos", "position")
    sib_xpath = _safe_get_attr(ins_action, "sibling_xpath")
    el = _safe_get_attr(ins_action, "node")
    if isinstance(el, _Element):
        try:
            if parent_xpath:
                parent = _get_element_by_xpath_from_tree(parent_xpath, mod_doc)
                if parent is not None:
                    target_idx = None
                    target_text = _get_normalized_full_text(el)[:40]
                    for i, ch in enumerate(list(parent)):
                        if getattr(ch, "tag", None) == getattr(el, "tag", None):
                            if _get_normalized_full_text(ch)[:40] == target_text:
                                target_idx = i
                                break
                    if isinstance(pos, int) or (isinstance(pos, str) and pos.isdigit()):
                        idx = int(pos)
                    elif target_idx is not None:
                        idx = target_idx
                    else:
                        idx = len(parent)
                    return el, parent_xpath, idx
        except Exception:
            pass

    if parent_xpath:
        parent = _get_element_by_xpath_from_tree(parent_xpath, mod_doc)
        if parent is not None:
            try:
                if isinstance(pos, int) or (isinstance(pos, str) and str(pos).isdigit()):
                    return parent[int(pos)], parent_xpath, int(pos)
                if pos in ("before", "after") and sib_xpath:
                    sib = _get_element_by_xpath_from_tree(sib_xpath, mod_doc)
                    if sib is not None:
                        par = sib.getparent()
                        idx = par.index(sib) + (1 if pos == "after" else 0)
                        try:
                            return par[idx], par.getroottree().getpath(par), idx
                        except Exception:
                            return None, par.getroottree().getpath(par), idx
                idx = len(parent)
                try:
                    return parent[idx], parent_xpath, idx
                except Exception:
                    return None, parent_xpath, idx
            except Exception:
                return None, parent_xpath, None

    if sib_xpath:
        sib = _get_element_by_xpath_from_tree(sib_xpath, mod_doc)
        if sib is not None:
            par = sib.getparent()
            idx = (
                par.index(sib) + (1 if pos == "after" else 0)
                if pos in ("before", "after")
                else len(par)
            )
            try:
                return par[idx], par.getroottree().getpath(par), idx
            except Exception:
                return None, par.getroottree().getpath(par), idx

    return None, None, None


def _score_pair(a: _NodeFP, b: _NodeFP) -> float:
    seq = difflib.SequenceMatcher(None, a.text_norm, b.text_norm, autojunk=False).ratio()
    jac = len(a.shingles & b.shingles) / max(1, len(a.shingles | b.shingles))
    content = max(seq, jac)
    shs = _simhash_sim(a.simhash, b.simhash)
    anc = difflib.SequenceMatcher(None, a.anchor_sig, b.anchor_sig, autojunk=False).ratio()
    attr = (
        1.0
        if (a.attrs_sig and a.attrs_sig == b.attrs_sig)
        else (0.5 if a.attrs_sig and b.attrs_sig else 0.0)
    )
    tag_bonus = (
        0.05
        if (a.tag == b.tag)
        else (0.03 if (a.tag in _BLOCK_TAGS and b.tag in _BLOCK_TAGS) else -0.10)
    )
    score = 0.45 * content + 0.25 * shs + 0.20 * anc + 0.10 * attr + tag_bonus
    lr = min(a.length, b.length) / max(a.length, b.length)
    if lr < 0.5:
        score *= 0.85
    return max(0.0, min(1.0, score))


def _augment_actions_with_fuzzy_moves(
    actions: List[Any],
    orig_doc: _ElementTree,
    mod_doc: _ElementTree,
    *,
    threshold: float = _MOVE_SIM_THRESHOLD,
    min_tokens: int = _MOVE_MIN_TOKENS,
    min_chars: int = _MOVE_MIN_CHARS,
    max_pairs: int = _MOVE_MAX_CANDIDATES,
) -> List[Any]:
    """Pair DeleteNode + InsertNode actions that represent moves and emit MoveNode."""
    if not actions:
        return actions

    del_fps: List[_NodeFP] = []
    ins_fps: List[_NodeFP] = []

    for idx, a in enumerate(actions):
        if isinstance(a, DeleteNode):
            xp = _safe_get_attr(a, "node", "node_xpath", "target")
            if not xp:
                continue
            el = _get_element_by_xpath_from_tree(xp, orig_doc)
            fp = _build_fp_for_element(el, xp)
            if fp:
                fp.action_index = idx
                del_fps.append(fp)

        elif isinstance(a, InsertNode):
            el, parent_xpath, pos = _resolve_insert_element(a, mod_doc)
            target_el = el or _get_element_by_xpath_from_tree(parent_xpath or "", mod_doc)
            if target_el is None:
                continue
            if el is None:
                node_struct = _safe_get_attr(a, "node")
                if isinstance(node_struct, _Element):
                    target_el = node_struct
            xp = None
            try:
                if el is not None:
                    xp = el.getroottree().getpath(el)
                elif parent_xpath is not None:
                    xp = parent_xpath
            except Exception:
                xp = parent_xpath or ""
            fp = _build_fp_for_element(target_el, xp or "")
            if fp:
                fp.parent_xpath = parent_xpath
                fp.pos = (
                    pos
                    if (isinstance(pos, int) or (isinstance(pos, str) and str(pos).isdigit()))
                    else None
                )
                fp.action_index = idx
                ins_fps.append(fp)

    if not del_fps or not ins_fps:
        return actions

    cand: List[Tuple[float, int, int]] = []
    for d in del_fps:
        for i in ins_fps:
            if len(cand) >= max_pairs:
                break
            if (d.tag != i.tag) and not (d.tag in _BLOCK_TAGS and i.tag in _BLOCK_TAGS):
                continue
            len_ratio = min(d.length, i.length) / max(d.length, i.length)
            if len_ratio < 0.40:
                continue
            if _simhash_sim(d.simhash, i.simhash) < 0.60:
                continue
            s = _score_pair(d, i)
            if s >= threshold:
                cand.append((s, d.action_index, i.action_index))
        if len(cand) >= max_pairs:
            break

    if not cand:
        return actions

    cand.sort(reverse=True)
    matched_d: Set[int] = set()
    matched_i: Set[int] = set()
    inject_moves: Dict[int, List[Any]] = {}

    def _mk_move_id(src_xp: str, tgt_px: str, pos_val: Optional[int]) -> str:
        h = hashlib.blake2b(
            f"{src_xp}|{tgt_px}|{pos_val}".encode("utf-8", "replace"), digest_size=6
        ).hexdigest()
        return f"mv_{h}"

    for score, d_idx, i_idx in cand:
        if d_idx in matched_d or i_idx in matched_i:
            continue
        d_act = actions[d_idx]
        i_act = actions[i_idx]
        src_xpath = _safe_get_attr(d_act, "node", "node_xpath", "target")
        tgt_parent = _safe_get_attr(i_act, "parent_xpath", "target")
        pos = _safe_get_attr(i_act, "pos", "position")
        sib_xpath = _safe_get_attr(i_act, "sibling_xpath")

        if isinstance(pos, str) and pos in ("before", "after") and sib_xpath:
            try:
                sib = _get_element_by_xpath_from_tree(sib_xpath, mod_doc)
                if sib is not None:
                    par = sib.getparent()
                    tgt_parent = par.getroottree().getpath(par)
                    pos = int(par.index(sib) + (1 if pos == "after" else 0))
            except Exception:
                pass
        elif not (isinstance(pos, int) or (isinstance(pos, str) and str(pos).isdigit())):
            try:
                parent_el = (
                    _get_element_by_xpath_from_tree(tgt_parent, mod_doc) if tgt_parent else None
                )
                if parent_el is not None:
                    pos = len(parent_el)
            except Exception:
                pos = "into"

        move_id = _mk_move_id(
            str(src_xpath), str(tgt_parent), int(pos) if str(pos).isdigit() else -1
        )
        try:
            if isinstance(pos, int) or (isinstance(pos, str) and str(pos).isdigit()):
                pos_int = int(pos)
                mv = MoveNode(src_xpath, tgt_parent, pos_int, move_id)  # type: ignore[call-arg]
            else:
                mv = MoveNode(src_xpath, tgt_parent, "into", move_id)  # type: ignore[call-arg]
        except TypeError:
            mv = MoveNode(node=src_xpath, target=tgt_parent, pos=pos, move_id=move_id)  # type: ignore[call-arg]

        inject_moves.setdefault(i_idx, []).append(mv)
        matched_d.add(d_idx)
        matched_i.add(i_idx)
        logger.debug(
            f"FuzzyMove paired del#{d_idx} -> ins#{i_idx} (score={score:.3f}, id={move_id})"
        )

    if not inject_moves:
        return actions

    new_actions: List[Any] = []
    for idx, act in enumerate(actions):
        if idx in matched_d:
            continue
        if idx in matched_i:
            for mv in inject_moves.get(idx, []):
                new_actions.append(mv)
            continue
        new_actions.append(act)

    logger.info(
        f"FuzzyMove: converted {len(matched_i)} insert/delete pairs into {sum(len(v) for v in inject_moves.values())} moves."
    )
    return new_actions


# ─────────────────────────────────────────────────────────────────────────────
#                               XSLT template
# ─────────────────────────────────────────────────────────────────────────────
_XMLDIFF_XSLT_REVISED = """<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:diff="http://namespaces.shoobx.com/diff"
    exclude-result-prefixes="diff">
  <!-- Removed xmlns:json and the comment -->
  <xsl:output method="html" omit-xml-declaration="yes" indent="no"/>
  <xsl:param name="diff-ns-uri" select="'http://namespaces.shoobx.com/diff'"/>

  <!-- Match all nodes and attributes, copy them -->
  <xsl:template match="@*|node()">
    <xsl:copy>
      <xsl:apply-templates select="@*|node()"/>
    </xsl:copy>
  </xsl:template>

  <!-- Template for inserted elements -->
  <xsl:template match="*[@diff:op='insert']">
    <ins class="diff-insert">
      <xsl:call-template name="copy-element-content"/>
    </ins>
  </xsl:template>

  <!-- Template for move target elements -->
  <xsl:template match="*[@diff:op='move-target']">
    <ins class="diff-move-target" data-move-id="{@diff:move-id}">
      <xsl:call-template name="copy-element-content"/>
    </ins>
  </xsl:template>

  <!-- Template for deleted elements -->
  <xsl:template match="*[@diff:op='delete']">
    <del class="diff-delete">
      <xsl:call-template name="copy-element-content"/>
    </del>
  </xsl:template>

  <!-- Template for move source elements -->
  <xsl:template match="*[@diff:op='move-source']">
    <del class="diff-move-source" data-move-id="{@diff:move-id}">
      <xsl:call-template name="copy-element-content"/>
    </del>
  </xsl:template>

    <!-- Template for elements with updated text/tail -->
    <xsl:template match="*[@diff:op='update-text' or
                        @diff:op='update-tail' or
                        @diff:op='update-text-context']">
    <span class="diff-update-container">
        <xsl:attribute name="title">
        <xsl:text>Original: </xsl:text>
        <xsl:value-of select="@diff:old-value"/>
        <xsl:if test="@diff:op='update-text-context'">
            <xsl:text> (</xsl:text>
            <xsl:value-of select="@diff:detail"/>
            <xsl:text>)</xsl:text>
        </xsl:if>
        </xsl:attribute>
        <xsl:call-template name="copy-element-content"/>
    </span>
    </xsl:template>

  <!-- Template for elements with attribute changes (if not already handled by insert/delete/move/text) -->
  <!-- Increased priority to override base copy if only attribs changed -->
  <xsl:template match="*[@diff:op='update-attrib']" priority="2">
     <!-- Check if the node ALSO has a major op; if so, let that template handle the wrapper -->
     <xsl:choose>
        <xsl:when test="@diff:op='insert' or @diff:op='delete' or @diff:op='move-target' or @diff:op='move-source' or starts-with(@diff:op, 'update-text') or @diff:op='rename-node'">
            <!-- Already handled by a more specific template, just copy content -->
             <xsl:call-template name="copy-element-content"/>
        </xsl:when>
        <xsl:otherwise>
            <!-- Only attribute changes, wrap in span -->
            <span class="diff-attrib-change">
              <xsl:attribute name="title">
                <xsl:call-template name="format-attribute-changes">
                  <xsl:with-param name="changes" select="@diff:attributes"/>
                </xsl:call-template>
              </xsl:attribute>
              <xsl:call-template name="copy-element-content"/>
            </span>
        </xsl:otherwise>
     </xsl:choose>
  </xsl:template>

 <xsl:template match="*[@diff:op='rename-node']">
    <span class="diff-rename-node">
      <xsl:attribute name="title">Renamed from <<xsl:value-of select="@diff:old-value"/>> to <<xsl:value-of select="@diff:new-value"/>></xsl:attribute>
      <xsl:call-template name="copy-element-content"/>
    </span>
  </xsl:template>

  <!-- Helper template to copy element content excluding diff attributes -->
  <xsl:template name="copy-element-content">
    <xsl:element name="{name()}" namespace="{namespace-uri()}">
      <!-- Copy non-diff attributes -->
      <xsl:apply-templates select="@*[not(namespace-uri()=$diff-ns-uri)]"/>
      <!-- Recursively apply templates to child nodes -->
      <xsl:apply-templates select="node()"/>
    </xsl:element>
  </xsl:template>

  <!-- Helper template for attribute changes (basic display for XSLT 1.0) -->
  <xsl:template name="format-attribute-changes">
    <xsl:param name="changes"/>
    <xsl:text>Attrs changed: </xsl:text>
    <!-- XSLT 1.0 cannot parse JSON. Display raw string. -->
    <xsl:value-of select="$changes"/>
  </xsl:template>

   <!-- Handle inline text changes explicitly -->
  <xsl:template match="ins[@class='diff-insert-text'] | del[@class='diff-delete-text']">
    <xsl:copy-of select="."/>
  </xsl:template>

  <!-- Prevent diff:* attributes from being copied to the output -->
  <xsl:template match="@diff:*" priority="10"/>

</xsl:stylesheet>"""


# ─────────────────────────────────────────────────────────────────────────────
#                               Public tool
# ─────────────────────────────────────────────────────────────────────────────


@with_tool_metrics
@with_error_handling
async def create_html_redline(
    original_html: str,
    modified_html: str,
    *,
    detect_moves: bool = True,
    formatting_tags: Optional[List[str]] = None,
    ignore_whitespace: bool = True,
    include_css: bool = True,
    add_navigation: bool = True,
    output_format: str = "html",
    use_tempfiles: bool = False,
    run_tidy: bool = False,
    generate_markdown: bool = False,
    markdown_path: str = "detected_redline_differences.md",
) -> Dict[str, Any]:
    """Generate a redline HTML comparing two HTML documents."""
    global _id_counter  # <-- Declare global at the top of the function scope
    t0 = time.time()
    logger.info("Starting HTML redline generation...")

    # --- Input Validation ---
    if not original_html or not isinstance(original_html, str):
        raise ToolInputError("original_html required")
    if not modified_html or not isinstance(modified_html, str):
        raise ToolInputError("modified_html required")
    if output_format not in {"html", "fragment"}:
        raise ToolInputError("output_format must be 'html' | 'fragment'")

    # --- Initialization for variables used outside try ---
    orig_tree: Optional[_ElementTree] = None
    mod_tree: Optional[_ElementTree] = None
    original_tree_pristine: Optional[_ElementTree] = None
    modified_tree_pristine: Optional[_ElementTree] = None

    logger.debug("Preprocessing HTML documents...")
    try:
        # --- Preprocessing ---
        orig_root, mod_root = _preprocess_html_docs(
            original_html,
            modified_html,
            ignore_whitespace=ignore_whitespace,
            use_tempfiles=use_tempfiles,
            run_tidy=run_tidy,
        )
        if orig_root is None or mod_root is None:
            raise ToolInputError("Preprocessing failed to return root elements.")

        # --- Reset Counter & Inject IDs ---
        logger.debug("Resetting ID counter and injecting synthetic IDs...")
        _id_counter = itertools.count(1)  # Reset counter here
        _inject_synthetic_ids(orig_root)  # Inject into original root
        _inject_synthetic_ids(mod_root)  # Inject into modified root
        logger.debug("Synthetic ID injection complete.")

        # --- Create Trees & Pristine Copies ---
        orig_tree = etree.ElementTree(orig_root)
        mod_tree = etree.ElementTree(mod_root)

        logger.debug("Creating pristine copies with IDs for formatter...")
        original_tree_pristine = deepcopy(orig_tree)
        modified_tree_pristine = deepcopy(mod_tree)
        logger.debug("Pristine copies created.")

    except Exception as e:
        logger.exception("Preprocessing, ID injection, or copying failed.")
        # Ensure pristine trees are None if we failed before creating them
        original_tree_pristine = None
        modified_tree_pristine = None
        raise ToolInputError("Failed HTML preparation") from e

    # --- Check if pristine copies were successfully created before proceeding ---
    if original_tree_pristine is None or modified_tree_pristine is None:
        # This case should ideally be caught by the exception above,
        # but it's good practice to check.
        logger.error("Pristine trees for diffing are missing after preparation step.")
        return {
            "redline_html": "<!-- Error: Failed to prepare documents for diffing -->",
            "stats": {"error": "Document preparation failed"},
            "processing_time": time.time() - t0,
            "success": False,
        }

    # --- Diff Actions (Using Synthetic IDs) ---
    logger.debug("Calculating differences using xmldiff with synthetic IDs...")
    # Options for the xmldiff Differ class constructor
    differ_opts: Dict[str, Any] = {
        "ratio_mode": "accurate",
        "fast_match": False,
        "F": 0.6,
        "uniqueattrs": ["data-diff-id"],
    }

    actions: List[Any] = []
    stats: Dict[str, Any] = {}
    markdown_summary = ""
    annotated_tree: Optional[_ElementTree] = None

    # --- Add this debug block ---
    try:
        debug_orig_path = "debug_orig_tree_with_ids.xml"
        debug_mod_path = "debug_mod_tree_with_ids.xml"
        with open(debug_orig_path, "wb") as f:
            orig_tree.write(f, pretty_print=True, encoding="utf-8", xml_declaration=True)
        with open(debug_mod_path, "wb") as f:
            mod_tree.write(f, pretty_print=True, encoding="utf-8", xml_declaration=True)
        logger.info(f"Debug trees with IDs written to {debug_orig_path} and {debug_mod_path}")
    except Exception as dbg_e:
        logger.warning(f"Failed to write debug trees: {dbg_e}")
    # --- End of debug block ---

    try:
        # Pass the trees WITH IDs to the diff engine
        # Use the main trees (orig_tree, mod_tree) for diffing
        # as they have the structure and IDs needed for diff calculation.
        actions = main.diff_trees(
            orig_tree,  # Use the tree derived directly from preprocessing + ID injection
            mod_tree,  # Use the tree derived directly from preprocessing + ID injection
            diff_options=differ_opts,
        )
        logger.info(f"xmldiff generated {len(actions)} actions using synthetic IDs.")

        # --- Fuzzy move augmentation (Delete+Insert -> MoveNode) ---
        try:
            actions = _augment_actions_with_fuzzy_moves(
                actions,
                original_tree_pristine,
                modified_tree_pristine,
                threshold=_MOVE_SIM_THRESHOLD,
                min_tokens=_MOVE_MIN_TOKENS,
                min_chars=_MOVE_MIN_CHARS,
                max_pairs=_MOVE_MAX_CANDIDATES,
            )
            logger.info(
                f"After FuzzyMove: {len(actions)} actions (moves inflated, deletes/inserts deflated)."
            )
        except Exception as e:
            logger.exception(f"Fuzzy move augmentation failed: {e}")

        # Debug: Log first few actions
        if actions:
            logger.debug(f"First 5 actions generated: {actions[:5]}")
        else:
            logger.warning("xmldiff generated NO actions.")

        # Check insert/delete ratio
        insert_delete_ratio = (
            (sum(1 for a in actions if isinstance(a, (InsertNode, DeleteNode))) / len(actions))
            if actions
            else 0
        )
        if insert_delete_ratio > 0.9:
            logger.warning(
                f"High ratio ({insert_delete_ratio:.2f}) of Insert/Delete actions. Node matching via data-diff-id might have failed."
            )

        # --- Generate Markdown Summary (if requested) ---
        if generate_markdown and actions:
            logger.debug("Generating Markdown summary...")
            # Pass the pristine copies (which include IDs for XPath lookup within the summary generation)
            markdown_summary = _generate_markdown_summary(
                orig_doc=original_tree_pristine, mod_doc=modified_tree_pristine, actions=actions
            )
            # (Consider where to save/return markdown_path content if needed)

        # --- Apply Actions using Formatter ---
        logger.debug("Applying actions using RedlineXMLFormatter...")
        formatter = RedlineXMLFormatter(
            detect_moves=detect_moves,
            normalize=formatting.WS_BOTH if ignore_whitespace else formatting.WS_NONE,
        )
        # Pass the pristine copies WITH IDs to the formatter.
        # The formatter needs the original pristine tree (with IDs) to find nodes
        # referenced by actions, and it works on a *copy* of this pristine tree.
        # It also needs the modified pristine tree for lookups (e.g., finding move targets).
        annotated_tree = formatter.format(actions, original_tree_pristine, modified_tree_pristine)
        stats = formatter.processed_actions
        logger.debug(f"Formatting complete. Stats: {stats}")

    except ToolError as te:
        logger.error(f"Diff engine error: {te}", exc_info=True)
        return {
            "redline_html": f"<!-- Diff Engine Error: {html_stdlib.escape(str(te))} -->",
            "stats": {"error": str(te)},
            "processing_time": time.time() - t0,
            "success": False,
        }
    except Exception as e:
        logger.exception("Error during diff/formatting.")
        return {
            "redline_html": "<!-- Error during diff/formatting -->",
            "stats": {"error": str(e)},
            "processing_time": time.time() - t0,
            "success": False,
        }
    finally:
        # Clear potentially large list to free memory
        actions.clear()
        # Explicitly None out large trees if possible (though garbage collection should handle this)
        orig_tree = mod_tree = original_tree_pristine = modified_tree_pristine = None

    # --- Remove Synthetic IDs from Final Output ---
    if annotated_tree is not None:
        logger.debug("Removing synthetic IDs from the final annotated tree...")
        count_removed = 0
        for el in annotated_tree.iter():
            if isinstance(el, _Element) and el.attrib.pop("data-diff-id", None):
                count_removed += 1
        logger.debug(f"Removed {count_removed} synthetic IDs from final output.")
    else:
        logger.error("Annotated tree is None after formatting.")
        # Handle this case - perhaps return an error
        return {
            "redline_html": "<!-- Error: Formatting produced no result -->",
            "stats": stats if stats else {"error": "Formatting failed"},
            "processing_time": time.time() - t0,
            "success": False,
        }

    # --- Apply XSLT ---
    logger.debug("Applying revised XSLT transformation...")
    redline_html = "<!-- XSLT Transformation Failed -->"
    # (Keep existing XSLT logic, ensuring annotated_tree is checked)
    try:
        xslt_root = etree.fromstring(_XMLDIFF_XSLT_REVISED.encode())
        transform = etree.XSLT(xslt_root)
        redline_doc = transform(annotated_tree)
        if redline_doc.getroot() is not None:
            redline_html = etree.tostring(
                redline_doc, encoding="unicode", method="html", pretty_print=False
            )
            logger.debug("XSLT transformation successful.")
        else:
            logger.error("XSLT transformation resulted in an empty document.")
            redline_html = "<!-- XSLT empty result -->"
    except Exception as e:
        logger.exception("XSLT transformation failed.")
        redline_html = f"<!-- XSLT Error: {html_stdlib.escape(str(e))} -->"

    # --- Post-processing ---
    logger.debug("Post-processing HTML output...")
    final_redline_html = await _postprocess_redline(
        redline_html,
        include_css=include_css,
        add_navigation=add_navigation,
        output_format=output_format,
    )
    logger.debug("Post-processing complete.")

    # --- Final Result ---
    dt = time.time() - t0
    success_flag = (
        stats.get("errors", 0) == 0
        and "<!-- XSLT" not in redline_html  # Check for XSLT error comments
    )
    result: Dict[str, Any] = {"stats": stats, "processing_time": dt, "success": success_flag}

    # Handle large output
    size_bytes = len(final_redline_html.encode("utf-8", errors="ignore"))
    logger.info(f"Generated redline HTML size: {size_bytes / 1024:.2f} KB")
    if size_bytes > 10_000_000:  # Example limit: 10MB
        logger.warning(f"Redline HTML size ({size_bytes} bytes) exceeds limit, encoding Base64.")
        try:
            result["redline_html_base64"] = base64.b64encode(
                final_redline_html.encode("utf-8")
            ).decode("ascii")
            result["output_is_base64"] = True  # Add flag
        except Exception as e:
            logger.error(f"Base64 encoding failed: {e}")
            result["redline_html"] = "<!-- Error: Output too large & Base64 failed -->"
            result["success"] = False
        # Avoid keeping large string in memory if encoded
        del final_redline_html
    else:
        result["redline_html"] = final_redline_html
        result["output_is_base64"] = False

    if generate_markdown:
        result["markdown_summary"] = markdown_summary
        if markdown_path:
            result["markdown_path"] = str(
                Path(markdown_path).resolve()
            )  # Example of returning path

    logger.info(
        f"HTML redline generation finished in {dt:.3f} seconds. Success: {result['success']}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
#                           Pre‑processing helpers
# ─────────────────────────────────────────────────────────────────────────────
def _check_tidy_available():
    try:
        res = subprocess.run(
            ["tidy", "--version"], capture_output=True, timeout=1, check=False, text=True
        )
        return res.returncode == 0 and "HTML Tidy" in res.stdout
    except Exception:
        return False


def _run_html_tidy(html: str) -> str:
    tidied_html = html
    with tempfile.TemporaryDirectory() as td:
        infile = Path(td, "input.html")
        infile.write_text(html, encoding="utf-8")
        cmd = [
            "tidy",
            "-q",
            "-m",
            "--tidy-mark",
            "no",
            "--drop-empty-elements",
            "no",
            "--wrap",
            "0",
            "--show-warnings",
            "no",
            "--show-errors",
            "0",
            "--force-output",
            "yes",
            "-utf8",
            str(infile),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)  # noqa: F841
            tidied_html = infile.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Tidy failed: {e}")
    return tidied_html


def _normalize_tree_whitespace(root: _Element) -> None:
    """Normalizes whitespace in .text and .tail for all elements in the tree in-place."""
    if root is None:
        return
    # Iterate through all elements AND comments/PIs (which can have tails)
    for node in root.xpath(". | .//node()"):
        # Check if it's an element or something else that can have text/tail
        if hasattr(node, "text"):
            node.text = _normalize_text(node.text) or None  # Use None if empty after normalize
        if hasattr(node, "tail"):
            node.tail = _normalize_text(node.tail) or None  # Use None if empty after normalize


def _deduplicate_body(root: _Element) -> None:
    """If an <html> element has more than one <body>, merge children
    into the first and delete the rest. Modifies the tree in-place."""
    if root is None or root.tag.lower() != "html":
        # Only operate on the root <html> element
        return

    bodies = root.xpath("./body | ./BODY")  # Case-insensitive check
    if len(bodies) <= 1:
        return  # Nothing to do

    logger.warning(f"Found {len(bodies)} <body> elements; merging into the first.")
    main_body = bodies[0]
    parent = main_body.getparent()  # Should be the <html> tag
    if parent is None:
        return  # Should not happen

    for i, extra_body in enumerate(bodies[1:], start=1):
        # Move children
        for child in list(extra_body):  # Iterate over a list copy
            main_body.append(child)  # Append moves the child

        # Append tail text if any
        if extra_body.tail:
            # Find the last element in main_body to append the tail to,
            # or append to main_body's text if it's empty
            last_element = main_body[-1] if len(main_body) > 0 else None
            if last_element is not None:
                if last_element.tail:
                    last_element.tail = (last_element.tail or "") + (extra_body.tail or "")
                else:
                    last_element.tail = extra_body.tail
            else:  # If main_body has no children, append to its text
                main_body.text = (main_body.text or "") + (extra_body.tail or "")

        # Remove the now-empty extra body
        try:
            parent.remove(extra_body)
        except ValueError:
            logger.error(f"Could not remove extra body #{i + 1}, already removed?")

    logger.debug("Finished merging duplicate <body> elements.")


def _preprocess_html_docs(
    original_html: str,
    modified_html: str,
    *,
    ignore_whitespace: bool = True,  # Keep this param, but handle normalization separately now
    use_tempfiles: bool = False,
    run_tidy: bool = False,
) -> Tuple[_Element, _Element]:
    """Preprocesses HTML, including optional Tidy and robust whitespace normalization."""

    if not original_html.strip():
        original_html = "<html><body><p>Empty Document</p></body></html>"  # Provide some structure
    if not modified_html.strip():
        modified_html = "<html><body><p>Empty Document</p></body></html>"  # Provide some structure

    tidied_orig, tidied_mod = original_html, modified_html

    # 1. Optional Tidy (Run *before* parsing)
    if run_tidy:
        logger.debug("Running HTML Tidy...")
        if _check_tidy_available():
            try:
                tidied_orig = _run_html_tidy(original_html)
                tidied_mod = _run_html_tidy(modified_html)
                logger.debug("HTML Tidy completed.")
            except Exception as e:
                logger.warning(f"HTML Tidy failed: {e}. Proceeding without Tidy.")
        else:
            logger.warning("HTML Tidy requested but not available. Skipping.")

    # 2. Parse HTML (Crucially, DO NOT remove blank text here initially)
    logger.debug("Parsing HTML documents with lxml...")
    parser = lxml_html.HTMLParser(
        recover=True,
        encoding="utf-8",
        remove_comments=False,  # Keep comments, they can affect structure/diff
        remove_pis=False,  # Keep processing instructions
        remove_blank_text=False,  # IMPORTANT: Keep blank text for now
    )
    o_root: Optional[_Element] = None
    m_root: Optional[_Element] = None
    try:
        # Use memory parsing unless very large docs require temp files
        if use_tempfiles and (len(tidied_orig) > 5e6 or len(tidied_mod) > 5e6):  # 5MB limit example
            logger.debug("Using temporary files for parsing large documents.")
            with tempfile.TemporaryDirectory() as td:
                orig_p = Path(td, "orig.html")
                mod_p = Path(td, "mod.html")
                orig_p.write_text(tidied_orig, encoding="utf-8")
                mod_p.write_text(tidied_mod, encoding="utf-8")
                o_root = lxml_html.parse(str(orig_p), parser=parser).getroot()
                m_root = lxml_html.parse(str(mod_p), parser=parser).getroot()
        else:
            # Ensure bytes for fromstring
            o_root = lxml_html.fromstring(tidied_orig.encode("utf-8"), parser=parser)
            m_root = lxml_html.fromstring(tidied_mod.encode("utf-8"), parser=parser)

        if o_root is None or m_root is None:
            raise ToolInputError("HTML parsing yielded None root element(s).")
        logger.debug("HTML parsing successful.")

    except Exception as e:
        logger.exception(f"HTML parsing failed: {e}")
        raise ToolInputError("Failed HTML parsing.") from e

    # 3. Normalize Whitespace (Apply *after* parsing)
    if ignore_whitespace:
        logger.debug("Normalizing whitespace in parsed trees...")
        try:
            _normalize_tree_whitespace(o_root)
            _normalize_tree_whitespace(m_root)
            logger.debug("Whitespace normalization complete.")
        except Exception as e:
            logger.exception("Whitespace normalization failed.")
            raise ToolInputError("Failed whitespace normalization during preprocessing.") from e

    # 4. Deduplicate Body Tags (Apply *after* normalization) <-- NEW STEP
    logger.debug("Checking for and merging duplicate <body> tags...")
    try:
        _deduplicate_body(o_root)
        _deduplicate_body(m_root)
        logger.debug("Duplicate <body> tag check complete.")
    except Exception as e:
        logger.exception("Failed during <body> deduplication.")
        # Decide whether to raise or just warn
        raise ToolInputError("Failed <body> tag deduplication during preprocessing.") from e

    # The roots returned now have a consistent whitespace representation
    # AND guaranteed single <body> element (if originally within <html>)
    return o_root, m_root


# ─────────────────────────────────────────────────────────────────────────────
#                       Post‑processing (CSS / nav UI)
# ─────────────────────────────────────────────────────────────────────────────
async def _postprocess_redline(
    redline_html: str,
    *,
    include_css: bool = True,
    add_navigation: bool = True,
    output_format: str = "html",
) -> str:
    if not redline_html or not redline_html.strip():
        return "<!-- Empty output -->"
    soup = BeautifulSoup(redline_html, "html.parser")
    if not soup.find("html", recursive=False):
        new_soup = BeautifulSoup(
            "<!DOCTYPE html><html><head><title>Comparison</title></head><body></body></html>",
            "html.parser",
        )
        if new_soup.body:
            [
                new_soup.body.append(deepcopy(el))
                for el in soup.contents
                if isinstance(el, Tag) or (isinstance(el, NavigableString) and el.strip())
            ]
        soup = new_soup
    html_tag = soup.html
    head = soup.head
    body = soup.body
    if not head:
        head = soup.new_tag("head")
        head.append(soup.new_tag("title", string="Comparison"))
        html_tag.insert(0, head)
    if not body:
        body = soup.new_tag("body")
        target = head.find_next_sibling() if head else None
        head.insert_after(body) if target else html_tag.append(body)
    if not head.find("meta", attrs={"name": "viewport"}):
        head.insert(
            0,
            soup.new_tag(
                "meta",
                attrs={"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
            ),
        )
    if not head.find("style", attrs={"data-base-diff": "1"}):
        head.append(BeautifulSoup(_get_base_diff_css(), "html.parser"))
    if include_css and not head.find("script", src=lambda s: s and "cdn.tailwindcss.com" in s):
        head.append(
            BeautifulSoup('<script src="https://cdn.tailwindcss.com"></script>', "html.parser")
        )
        if not head.find("link", href=lambda x: x and "fonts.googleapis.com" in x):
            [
                head.append(
                    BeautifulSoup(f'<link rel="preconnect" href="https://{u}"{a}>', "html.parser")
                )
                for u, a in [("fonts.googleapis.com", ""), ("fonts.gstatic.com", " crossorigin")]
            ]
            head.append(
                BeautifulSoup(
                    '<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,200..800;1,6..72,200..800&display=swap" rel="stylesheet">',
                    "html.parser",
                )
            )
        style_tag = soup.new_tag("style", type="text/tailwindcss")
        style_tag.string = _get_tailwind_css()
        head.append(style_tag)
    if add_navigation and output_format == "html":
        if not body.find("div", class_="redline-minimap"):
            body.append(
                BeautifulSoup(
                    """<div class="redline-minimap fixed right-1 top-10 bottom-10 w-1 bg-gray-100 dark:bg-gray-800 rounded z-40 hidden md:flex flex-col"></div>""",
                    "html.parser",
                )
            )
        if not body.find("div", class_="redline-navigation"):
            body.insert(
                0,
                BeautifulSoup(
                    """<div class="redline-navigation fixed top-2 right-2 bg-white/90 dark:bg-gray-800/90 p-2 rounded-lg shadow-lg z-50 text-xs backdrop-blur-sm"><div class="flex items-center"><button class="btn" onclick="goPrevChange()">Prev</button><button class="btn" onclick="goNextChange()">Next</button><span class="ml-2 px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded" id="change-counter">-/-</span></div></div>""".replace(
                        'class="btn"',
                        'class="bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 px-2 py-1 rounded mr-1 transition"',
                    )
                    .replace(
                        "Prev",
                        '<span class="hidden sm:inline">Previous</span><span class="sm:hidden">←</span>',
                    )
                    .replace(
                        "Next",
                        '<span class="hidden sm:inline">Next</span><span class="sm:hidden">→</span>',
                    ),
                    "html.parser",
                ),
            )
        if not body.find("div", class_="redline-legend"):
            body.append(
                BeautifulSoup(
                    """<div class="redline-legend fixed bottom-2 left-2 bg-white/90 dark:bg-gray-800/90 p-2 rounded-lg shadow-lg z-50 text-xs flex flex-wrap gap-2 backdrop-blur-sm"><span class="legend-item"><span class="legend-color bg-blue-100 ring-blue-300 dark:bg-blue-900/60 dark:ring-blue-700"></span>Insert</span><span class="legend-item"><span class="legend-color bg-rose-100 ring-rose-300 dark:bg-rose-900/60 dark:ring-rose-700"></span>Delete</span><span class="legend-item"><span class="legend-color bg-emerald-100 ring-emerald-300 dark:bg-emerald-900/60 dark:ring-emerald-700"></span>Move</span><span class="legend-item"><span class="legend-color bg-orange-100 ring-orange-300 dark:bg-orange-900/60 dark:ring-orange-700"></span>Attr</span></div>""".replace(
                        'class="legend-item"', 'class="flex items-center"'
                    ).replace(
                        'class="legend-color', 'class="inline-block w-3 h-3 rounded ring-1 mr-1'
                    ),
                    "html.parser",
                )
            )
        if not body.find("button", id="theme-toggle"):
            body.insert(
                1,
                BeautifulSoup(
                    """<button id="theme-toggle" title="Toggle theme" class="fixed top-2 left-2 z-50 p-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg text-xs"><svg class="h-4 w-4 hidden dark:inline" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"/></svg><svg class="h-4 w-4 dark:hidden" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"/></svg></button>""",
                    "html.parser",
                ),
            )
        if not body.find("script", string=lambda s: s and "goNextChange" in s):
            script_tag = soup.new_tag("script")
            script_tag.string = _get_navigation_js()
            body.append(script_tag)
    body_classes = body.get("class", [])
    req_cls = [
        "font-['Newsreader']",
        "dark:text-gray-200",
        "dark:bg-gray-900",
        "transition-colors",
        "duration-200",
    ]
    [body_classes.append(c) for c in req_cls if c not in body_classes]
    body["class"] = body_classes
    if output_format == "html" and not any(
        isinstance(c, Tag) and "prose" in c.get("class", []) for c in body.contents
    ):
        wrapper = soup.new_tag(
            "div", **{"class": "prose lg:prose-xl dark:prose-invert mx-auto max-w-4xl px-4 py-8"}
        )
        ui_els = body.select(
            ".redline-navigation, .redline-legend, .redline-minimap, #theme-toggle, script",
            recursive=False,
        )
        content = [el for el in body.contents if el not in ui_els]
        [
            wrapper.append(el.extract())
            for el in content
            if isinstance(el, Tag) or (isinstance(el, NavigableString) and el.strip())
        ]
        body.append(wrapper)
    final_html = body.decode_contents() if output_format == "fragment" else str(soup)
    logger.debug("HTML postprocessing finished.")
    return final_html


def _get_base_diff_css() -> str:
    # (Copied from previous correct version)
    return """
        <style data-base-diff="1">
          ins.diff-insert, .diff-move-target {color:#1d4ed8; background-color:#eff6ff; border:1px solid #93c5fd; padding:0 1px; margin:0 1px; border-radius:2px; text-decoration:none;}
          del.diff-delete, .diff-move-source {color:#b91c1c; background-color:#fef2f2; border:1px solid #fca5a5; padding:0 1px; margin:0 1px; border-radius:2px; text-decoration:line-through;}
          ins.diff-move-target {color:#047857; background-color:#ecfdf5; border:1px solid #6ee7b7;}
          del.diff-move-source {color:#065f46; background-color:#ecfdf599; border:1px dashed #6ee7b7; }
          span.diff-update-container > * {border-bottom: 1px dotted #f97316;}
          span.diff-attrib-change > * {box-shadow: 0px 0px 0px 1px #fb923c inset; }
          span.diff-rename-node > * {box-shadow: 0px 0px 0px 1px #a855f7 inset; }
          ins.diff-insert-text {color:#1e40af; text-decoration:underline; background:transparent; border:none; padding:0; margin:0;}
          del.diff-delete-text {color:#b91c1c; text-decoration:line-through; background:transparent; border:none; padding:0; margin:0;}
          /* Basic dark mode */
          @media (prefers-color-scheme: dark) {
            body { background-color: #1f2937; color: #d1d5db; }
            ins.diff-insert, .diff-move-target { color: #93c5fd; background-color: #1e3a8a; border-color: #3b82f6; }
            del.diff-delete, .diff-move-source { color: #fca5a5; background-color: #7f1d1d; border-color: #ef4444; }
            ins.diff-move-target { color: #6ee7b7; background-color: #065f46; border-color: #10b981; }
            del.diff-move-source { color: #a7f3d0; background-color: #064e3b; border-color: #34d399; }
            span.diff-update-container > * { border-color: #fb923c; }
            span.diff-attrib-change > * { box-shadow: 0px 0px 0px 1px #f97316 inset; }
            span.diff-rename-node > * { box-shadow: 0px 0px 0px 1px #c084fc inset; }
            ins.diff-insert-text {color:#60a5fa;}
            del.diff-delete-text {color:#f87171;}
          }
        </style>
        """


def _get_tailwind_css() -> str:
    return """ @tailwind base;@tailwind components;@tailwind utilities; @layer components { .diff-insert, .diff-delete, .diff-move-target, .diff-move-source { @apply px-0.5 rounded-sm mx-[1px] transition duration-150; } ins.diff-insert, .diff-insert > ins { @apply text-blue-800 bg-blue-50 ring-1 ring-inset ring-blue-300/60 no-underline; } .dark ins.diff-insert, .dark .diff-insert > ins { @apply text-blue-200 bg-blue-900/40 ring-blue-500/30; } ins.diff-insert:hover, .diff-insert > ins:hover { @apply ring-2 ring-offset-1 ring-black/10 shadow-sm bg-blue-100 dark:bg-blue-800/60; } del.diff-delete, .diff-delete > del { @apply text-rose-800 bg-rose-50 ring-1 ring-inset ring-rose-300/60 line-through; } .dark del.diff-delete, .dark .diff-delete > del { @apply text-rose-200 bg-rose-900/40 ring-rose-500/30; } del.diff-delete:hover, .diff-delete > del:hover { @apply ring-2 ring-offset-1 ring-black/10 shadow-sm bg-rose-100 dark:bg-rose-800/60; } ins.diff-move-target, .diff-move-target > ins { @apply text-emerald-900 bg-emerald-50 ring-1 ring-emerald-400/60 no-underline border border-emerald-300; } .dark ins.diff-move-target, .dark .diff-move-target > ins { @apply text-emerald-200 bg-emerald-900/40 ring-emerald-500/30 border-emerald-700; } ins.diff-move-target:hover, .diff-move-target > ins:hover { @apply ring-2 ring-offset-1 ring-black/10 shadow-sm bg-emerald-100 dark:bg-emerald-800/60; } del.diff-move-source, .diff-move-source > del { @apply text-emerald-800/60 bg-emerald-50/50 line-through border border-dashed border-emerald-400/40; } .dark del.diff-move-source, .dark .diff-move-source > del { @apply text-emerald-300/60 bg-emerald-900/30 border-emerald-700/40; } del.diff-move-source:hover, .diff-move-source > del:hover { @apply bg-emerald-100/70 border-emerald-400 shadow-sm dark:bg-emerald-800/50; } span.diff-update-container { @apply border-b border-dotted border-orange-400 bg-orange-50/30; } .dark span.diff-update-container { @apply border-orange-500 bg-orange-900/30; } span.diff-update-container:hover { @apply bg-orange-100/50 dark:bg-orange-800/40; } span.diff-attrib-change { @apply ring-1 ring-orange-400/50 ring-inset bg-orange-50/30 backdrop-blur-sm rounded-sm; } .dark span.diff-attrib-change { @apply ring-orange-500/50 bg-orange-900/30; } span.diff-attrib-change:hover { @apply bg-orange-100/50 dark:bg-orange-800/40; } span.diff-rename-node { @apply ring-1 ring-purple-400/50 ring-inset bg-violet-50/30 backdrop-blur-sm rounded-sm; } .dark span.diff-rename-node { @apply ring-purple-500/50 bg-violet-900/30; } span.diff-rename-node:hover { @apply bg-violet-100/50 dark:bg-violet-800/40; } ins.diff-insert-text { @apply text-blue-700 dark:text-blue-300 underline decoration-dotted decoration-1 underline-offset-2 bg-transparent border-none ring-0 p-0 m-0; } del.diff-delete-text { @apply text-rose-700 dark:text-rose-300 line-through decoration-dotted decoration-1 bg-transparent border-none ring-0 p-0 m-0; } @media print { .redline-navigation, .redline-legend, .redline-minimap, #theme-toggle { @apply hidden; } ins, del, span[class*="diff-"] { @apply text-black !important; background-color: transparent !important; border: none !important; ring: none !important; box-shadow: none !important; } ins { @apply font-bold no-underline; } del { @apply italic line-through; } } } """


def _get_navigation_js() -> str:
    return """ /* Combined JS */ document.addEventListener('DOMContentLoaded', () => { let _redlineChanges = null; let _changeIdx = -1; let _currentHi = null; const changeCounter = document.getElementById('change-counter'); const minimap = document.querySelector('.redline-minimap'); const container = document.querySelector('.prose') || document.body; function findAllChanges() { if (!_redlineChanges) { _redlineChanges = Array.from(container.querySelectorAll( 'ins.diff-insert, ins.diff-move-target, del.diff-delete, del.diff-move-source, span.diff-update-container, span.diff-attrib-change, span.diff-rename-node, ins.diff-insert-text, del.diff-delete-text' )); _redlineChanges = _redlineChanges.filter(el => { let p = el.parentElement; while (p && p !== container && p !== document.body) { if (_redlineChanges.includes(p)) return false; p = p.parentElement; } return true; }); _redlineChanges.sort((a, b) => { const c = a.compareDocumentPosition(b); if (c & Node.DOCUMENT_POSITION_FOLLOWING) return 1; if (c & Node.DOCUMENT_POSITION_PRECEDING) return -1; return 0; }); updateCounter(); if (_redlineChanges && _redlineChanges.length > 0) { _changeIdx = -1; } else { console.log("Redline Nav: No changes found."); } } return _redlineChanges; } function highlightCurrentChange(scrollTo = true) { if (!_redlineChanges || _changeIdx < 0 || _changeIdx >= _redlineChanges.length) return; const el = _redlineChanges[_changeIdx]; if (!el) return; if (_currentHi && _currentHi !== el && _redlineChanges.includes(_currentHi)) { _currentHi.style.outline = ''; _currentHi.style.boxShadow = ''; _currentHi.style.outlineOffset = ''; _currentHi.classList.remove('current-redline-change'); } el.style.outline = '2px solid orange'; el.style.outlineOffset = '2px'; el.style.boxShadow = '0 0 8px 1px rgba(255, 165, 0, 0.6)'; el.classList.add('current-redline-change'); if (scrollTo) { const rect = el.getBoundingClientRect(); const isVisible = rect.top >= 0 && rect.left >= 0 && rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) && rect.right <= (window.innerWidth || document.documentElement.clientWidth); if (!isVisible) { el.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' }); } } _currentHi = el; updateCounter(); updateMinimapHighlight(); } window.goPrevChange = () => { findAllChanges(); if (!_redlineChanges || _redlineChanges.length === 0) return; _changeIdx = (_changeIdx <= 0) ? _redlineChanges.length - 1 : _changeIdx - 1; highlightCurrentChange(); }; window.goNextChange = () => { findAllChanges(); if (!_redlineChanges || _redlineChanges.length === 0) return; _changeIdx = (_changeIdx >= _redlineChanges.length - 1) ? 0 : _changeIdx + 1; highlightCurrentChange(); }; function updateCounter() { if (changeCounter && _redlineChanges) { changeCounter.textContent = `${_redlineChanges.length > 0 ? _changeIdx + 1 : 0}/${_redlineChanges.length}`; } else if (changeCounter) { changeCounter.textContent = '0/0'; } } document.addEventListener("keydown", e => { if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) return; if (e.key === "ArrowRight" && !e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey) { goNextChange(); e.preventDefault(); } else if (e.key === "ArrowLeft" && !e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey) { goPrevChange(); e.preventDefault(); } }); let minimapMarkers = []; function setupMinimap() { if (!minimap || !container) { return; } const changes = findAllChanges(); if (!changes || !changes.length) { minimap.style.display = 'none'; return; } minimap.innerHTML = ''; minimapMarkers = []; requestAnimationFrame(() => { const containerHeight = container.scrollHeight; if (containerHeight <= 0) { return; } changes.forEach((change, index) => { let type = ''; const cl = change.classList; const tagName = change.tagName.toUpperCase(); if (cl.contains('diff-insert') || (tagName === 'INS' && !cl.contains('diff-move-target'))) type = 'insert'; else if (cl.contains('diff-delete') || (tagName === 'DEL' && !cl.contains('diff-move-source'))) type = 'delete'; else if (cl.contains('diff-move-target') || cl.contains('diff-move-source')) type = 'move'; else if (cl.contains('diff-attrib-change')) type = 'attrib'; else if (cl.contains('diff-update-container')) type = 'text'; else if (cl.contains('diff-rename-node')) type = 'rename'; else if (cl.contains('diff-insert-text')) type = 'insert'; else if (cl.contains('diff-delete-text')) type = 'delete'; else return; const relativePos = change.offsetTop / containerHeight; const marker = document.createElement('div'); marker.className = 'minimap-marker absolute w-full h-[3px] cursor-pointer opacity-75 hover:opacity-100 transition-opacity duration-150'; marker.style.top = `${Math.max(0, Math.min(100, relativePos * 100))}%`; if (type === 'insert') marker.classList.add('bg-blue-500'); else if (type === 'delete') marker.classList.add('bg-rose-500'); else if (type === 'move') marker.classList.add('bg-emerald-500'); else marker.classList.add('bg-orange-500'); marker.title = `${type.charAt(0).toUpperCase() + type.slice(1)} change (${index + 1}/${changes.length})`; marker.dataset.changeIndex = index; marker.addEventListener('click', () => { _changeIdx = index; highlightCurrentChange(); }); minimap.appendChild(marker); minimapMarkers.push(marker); }); minimap.style.display = 'flex'; updateMinimapHighlight(); }); } function updateMinimapHighlight() { minimapMarkers.forEach((marker, index) => { if (index === _changeIdx) { marker.style.transform = 'scaleX(1.5)'; marker.style.opacity = '1'; marker.style.zIndex = '10'; marker.classList.add('bg-yellow-400'); marker.classList.remove('bg-blue-500', 'bg-rose-500', 'bg-emerald-500', 'bg-orange-500'); } else { marker.style.transform = ''; marker.style.opacity = '0.75'; marker.style.zIndex = '1'; marker.classList.remove('bg-yellow-400'); const oClass = marker.title.includes('Insert') ? 'bg-blue-500' : marker.title.includes('Delete') ? 'bg-rose-500' : marker.title.includes('Move') ? 'bg-emerald-500' : 'bg-orange-500'; if (!marker.classList.contains(oClass)) { marker.classList.remove('bg-blue-500', 'bg-rose-500', 'bg-emerald-500', 'bg-orange-500'); marker.classList.add(oClass); } } }); } function debounce(func, wait) { let t; return function(...a) { const l = () => { clearTimeout(t); func(...a); }; clearTimeout(t); t = setTimeout(l, wait); }; } const debouncedSetupMinimap = debounce(setupMinimap, 250); window.addEventListener('resize', debouncedSetupMinimap); const themeToggle = document.getElementById('theme-toggle'); function applyTheme(isDark) { document.documentElement.classList.toggle('dark', isDark); localStorage.theme = isDark ? 'dark' : 'light'; setupMinimap(); } if (themeToggle) { const pDark = window.matchMedia('(prefers-color-scheme: dark)').matches; const cTheme = localStorage.theme === 'dark' || (!('theme' in localStorage) && pDark) ? 'dark' : 'light'; applyTheme(cTheme === 'dark'); themeToggle.addEventListener('click', () => { applyTheme(!document.documentElement.classList.contains('dark')); }); } function handleMoveHighlight(event) { const moveEl = event.target.closest("ins[data-move-id], del[data-move-id]"); if (!moveEl) return; const moveId = moveEl.dataset.moveId; if (!moveId) return; const isEnter = event.type === "mouseover"; document.querySelectorAll(`[data-move-id='${moveId}']`).forEach(el => { el.style.outline = isEnter ? "3px dashed #059669" : ""; el.style.outlineOffset = isEnter ? "2px" : ""; el.style.transition = 'outline 0.15s ease-in-out, outline-offset 0.15s ease-in-out'; }); } container.addEventListener("mouseover", handleMoveHighlight); container.addEventListener("mouseout", handleMoveHighlight); findAllChanges(); setupMinimap(); }); """


# ─────────────────────────────────────────────────────────────────────────────
#                       Plain‑text comparison (escaped)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_text_redline(
    original_text: str,
    modified_text: str,
    *,
    diff_level: str = "word",
) -> Tuple[str, Dict[str, int]]:
    """Return plain‑text diff with {- +} markers and [~ ~] for moves."""
    if diff_level == "char":
        orig_units, mod_units, joiner = list(original_text), list(modified_text), ""
    elif diff_level == "word":
        rx = r"(\w+[\S\w]*|\s+|[^\w\s])"  # Keep whitespace as separate unit
        orig_units, mod_units, joiner = (
            re.findall(rx, original_text),
            re.findall(rx, modified_text),
            "",
        )
    else:  # line level
        orig_units, mod_units, joiner = (
            original_text.splitlines(True),
            modified_text.splitlines(True),
            "",
        )

    sm = difflib.SequenceMatcher(None, orig_units, mod_units, autojunk=False)
    ops: List[Tuple[str, str]] = []  # Store ('tag', 'text') pairs
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            ops.append(("eq", joiner.join(orig_units[i1:i2])))
        elif tag == "delete":
            ops.append(("del", joiner.join(orig_units[i1:i2])))
        elif tag == "insert":
            ops.append(("ins", joiner.join(mod_units[j1:j2])))
        else:  # replace
            # Treat replace as delete followed by insert for move detection
            ops.append(("del", joiner.join(orig_units[i1:i2])))
            ops.append(("ins", joiner.join(mod_units[j1:j2])))

    # --- Attempt Move Detection based on Content ---
    # Normalize whitespace and ignore case for matching identical blocks
    def _normalize_for_match(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip()).lower()

    dels: Dict[str, List[int]] = {}  # Map normalized text -> list of deletion indices
    ins: Dict[str, List[int]] = {}  # Map normalized text -> list of insertion indices
    paired: Dict[int, int] = {}  # Map deletion index -> insertion index for identified moves

    for idx, (tag, txt) in enumerate(ops):
        if tag == "del":
            key = _normalize_for_match(txt)
            if key:  # Only track non-empty deletions
                dels.setdefault(key, []).append(idx)
        elif tag == "ins":
            key = _normalize_for_match(txt)
            if key:  # Only track non-empty insertions
                ins.setdefault(key, []).append(idx)

    # Find potential moves: identical normalized content deleted once and inserted once
    for key in set(dels) & set(ins):
        if len(dels[key]) == 1 and len(ins[key]) == 1:
            deletion_idx = dels[key][0]
            insertion_idx = ins[key][0]
            # Ensure they are not adjacent (which would be a replace)
            # This simple check might be too strict, but helps avoid marking simple replacements as moves
            if abs(deletion_idx - insertion_idx) > 1:
                paired[deletion_idx] = insertion_idx  # Mark as a move pair

    # --- Build Output String ---
    buf: List[str] = []
    ic = dc = mc = 0  # Insert, Delete, Move counts
    for idx, (tag, txt) in enumerate(ops):
        if idx in paired:  # This is the deletion part of a move, skip it
            continue
        if idx in paired.values():  # This is the insertion part of a move
            # Escape markers within the moved text
            escaped_move = txt.replace("[~", "[ ~").replace("~]", "~ ]")
            buf.append(f"[~{escaped_move}~]")
            mc += 1
            continue

        # Handle regular operations
        if tag == "eq":
            buf.append(txt)
        elif tag == "del":
            # Escape markers within the deleted text
            escaped_del = txt.replace("[-", "[ -").replace("-]", "- ]")
            buf.append(f"[-{escaped_del}-]")
            dc += 1
        elif tag == "ins":
            # Escape markers within the inserted text
            escaped_ins = txt.replace("{+", "{ +").replace("+}", "+ }")
            buf.append(f"{{+{escaped_ins}+}}")
            ic += 1

    # --- Calculate Stats ---
    stats = {
        "total_changes": ic + dc + mc,  # Total distinct changes
        "insertions": ic,
        "deletions": dc,
        "moves": mc,
        "text_updates": 0,  # Not explicitly tracked with this method
        "attr_updates": 0,
        "other_changes": 0,
        "inline_insertions": ic + mc,  # Count move insertions here?
        "inline_deletions": dc + mc,  # Count move deletions here?
    }
    return "".join(buf), stats


# ─────────────────────────────────────────────────────────────────────────────
#                       Public wrapper for text docs
# ─────────────────────────────────────────────────────────────────────────────
@with_tool_metrics
@with_error_handling
async def compare_documents_redline(
    original_text: str,
    modified_text: str,
    *,
    file_format: str = "auto",
    detect_moves: bool = True,
    ignore_whitespace: bool = True,
    output_format: str = "html",
    diff_level: str = "word",
    include_css: bool = True,
    generate_markdown: bool = False,
    markdown_path: str = "detected_redline_differences.md",
    run_tidy: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    logger.info(f"Starting doc comparison. Input: {file_format}, Output: {output_format}")
    if not isinstance(original_text, str):
        raise ToolInputError("original_text must be str")
    if not isinstance(modified_text, str):
        raise ToolInputError("modified_text must be str")
    valid_formats = {"auto", "html", "text", "markdown", "latex"}
    if file_format not in valid_formats:
        raise ToolInputError(f"Invalid file_format: {file_format}")
    if output_format not in {"html", "text"}:
        raise ToolInputError(f"Invalid output_format: {output_format}")
    if diff_level not in {"char", "word", "line"}:
        raise ToolInputError(f"Invalid diff_level: {diff_level}")

    if original_text == modified_text:
        logger.info("Documents are identical.")
        stats = {k: 0 for k in RedlineXMLFormatter().processed_actions}
        stats["total_changes"] = 0
        if output_format == "html":
            fmt = file_format if file_format != "auto" else _detect_file_format(original_text)
            html = ""
            try:
                if fmt == "html":
                    html = original_text
                elif fmt == "markdown":
                    md_ext = [
                        "fenced_code",
                        "tables",
                        "sane_lists",
                        "nl2br",
                        "footnotes",
                        "attr_list",
                    ]
                    html = markdown.markdown(original_text, extensions=md_ext)
                elif _DOC_CONVERSION_AVAILABLE:
                    res = await convert_document(
                        document_data=original_text.encode("utf-8"),
                        input_format_hint=fmt,
                        output_format="markdown",
                    )
                    if res.get("success") and res.get("content"):
                        md_ext = [
                            "fenced_code",
                            "tables",
                            "sane_lists",
                            "nl2br",
                            "footnotes",
                            "attr_list",
                        ]
                        html = markdown.markdown(res["content"], extensions=md_ext)
                    else:
                        logger.warning(f"Conv failed: {res.get('error')}")
                        html = f"<pre>{html_stdlib.escape(original_text)}</pre>"
                else:
                    html = f"<pre>{html_stdlib.escape(original_text)}</pre>"
                final_html = await _postprocess_redline(
                    html, include_css=True, add_navigation=False, output_format="html"
                )
            except Exception as e:
                logger.error(f"Error prep identical: {e}")
                final_html = f"<!DOCTYPE html><html><body><pre>{html_stdlib.escape(original_text)}</pre></body></html>"
            return {
                "redline_html": final_html,
                "stats": stats,
                "processing_time": time.time() - t0,
                "success": True,
            }
        else:
            return {
                "redline": original_text,
                "stats": stats,
                "processing_time": time.time() - t0,
                "success": True,
            }

    actual_format = file_format
    if actual_format == "auto":
        actual_format = _detect_file_format(original_text)
        logger.info(f"Auto-detected format: {actual_format}")

    if output_format == "html":
        logger.info(f"Generating HTML redline for '{actual_format}' input...")
        orig_html = original_text
        mod_html = modified_text
        if actual_format != "html":
            if not _DOC_CONVERSION_AVAILABLE:
                raise ToolError(
                    f"Input '{actual_format}', but conversion tool unavailable.",
                    code="DEPENDENCY_MISSING",
                )
            logger.info(f"Converting '{actual_format}' input to Markdown then HTML...")
            try:
                params = {
                    "output_format": "markdown",
                    "extraction_strategy": "hybrid_direct_ocr",
                    "enhance_with_llm": False,
                }
                res_o = await convert_document(
                    document_data=original_text.encode("utf-8"),
                    input_format_hint=actual_format,
                    **params,
                )
                if not res_o.get("success"):
                    raise ToolError(
                        f"Orig conv failed: {res_o.get('error')}", code="CONVERSION_FAILED"
                    )
                res_m = await convert_document(
                    document_data=modified_text.encode("utf-8"),
                    input_format_hint=actual_format,
                    **params,
                )
                if not res_m.get("success"):
                    raise ToolError(
                        f"Mod conv failed: {res_m.get('error')}", code="CONVERSION_FAILED"
                    )
                md_ext = ["fenced_code", "tables", "sane_lists", "nl2br", "footnotes", "attr_list"]
                orig_html = markdown.markdown(res_o["content"], extensions=md_ext)
                mod_html = markdown.markdown(res_m["content"], extensions=md_ext)
            except Exception as e:
                logger.error(f"Doc conversion failed: {e}", exc_info=True)
                raise ToolInputError("Failed doc conversion.") from e
        html_result = await create_html_redline(
            original_html=orig_html,
            modified_html=mod_html,
            detect_moves=detect_moves,
            ignore_whitespace=ignore_whitespace,
            output_format="html",
            include_css=include_css,
            add_navigation=True,
            generate_markdown=generate_markdown,
            markdown_path=markdown_path,
            run_tidy=run_tidy,
        )
        html_result["processing_time"] = time.time() - t0
        return html_result

    elif output_format == "text":
        logger.info(f"Generating plain text redline (level: {diff_level})...")
        o_plain, m_plain = original_text, modified_text
        if actual_format == "html":
            logger.warning("Generating text diff from HTML; tags included.")
        elif actual_format == "markdown":
            logger.warning("Generating text diff from Markdown; syntax included.")
        txt, stats = _generate_text_redline(o_plain, m_plain, diff_level=diff_level)
        return {
            "redline": txt,
            "stats": stats,
            "processing_time": time.time() - t0,
            "success": True,
        }

    raise ToolInputError("Invalid output format.")


# ─────────────────────────────────────────────────────────────────────────────
#                               Aux helpers
# ─────────────────────────────────────────────────────────────────────────────
def _detect_file_format(text: str) -> str:
    if not text or not text.strip():
        return "text"
    t = text.lower().strip()
    if t.startswith("<!doctype html") or t.startswith("<html"):
        return "html"
    hs = sum(f"<{tag}" in t for tag in ("body", "div", "p", "table", "h1", "br")) + sum(
        f"</{tag}>" in t for tag in ("body", "div", "p", "table", "h1")
    )
    lrx = [
        r"\\documentclass",
        r"\\begin\{document\}",
        r"\\section\{",
        r"\\usepackage\{",
        r"\$.+\$",
        r"\\begin\{",
    ]
    ls = sum(bool(re.search(p, text, re.M | re.I)) for p in lrx)
    mrx = [
        r"^[#]+\s+",
        r"^>\s+",
        r"^\s*[-*+]\s+",
        r"^\s*[0-9]+\.\s+",
        r"```|~~~",
        r"\|.*\|.*\|",
        r"\*{1,2}[^*\s]",
        r"`[^`]+`",
        r"\[.*?\]\(.*?\)",
    ]
    lines = text.splitlines()
    ms = (
        sum(bool(re.search(p, l)) for p in mrx[:5] for l in lines[:30])  # noqa: E741
        + sum(bool(re.search(p, text, re.M)) for p in mrx[5:])
    )
    if ls >= 2 and hs < 2:
        return "latex"
    if hs >= 4 or (hs >= 2 and "<body" in t):
        return "html"
    if ms >= 3 and hs <= 1 and ls <= 0:
        return "markdown"
    if hs >= 2 and ms >= 2:
        return "html"
    return "text"


# ─────────────────────────────────────────────────────────────────────────────
#                               Metadata
# ─────────────────────────────────────────────────────────────────────────────
__all__ = ["create_html_redline", "compare_documents_redline", "RedlineXMLFormatter"]
__version__ = "1.5.0"
__updated__ = _dt.datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"
