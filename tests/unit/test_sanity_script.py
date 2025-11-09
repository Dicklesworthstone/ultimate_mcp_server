import importlib.util
import sys
from dataclasses import dataclass
from typing import Any, Tuple

import pytest


@pytest.fixture(scope="module")
def sanity_module():
    """Dynamically import the top-level test_sanity.py as a module for unit testing."""
    path = "/home/ec2-user/git/ultimate_mcp_server/test_sanity.py"
    spec = importlib.util.spec_from_file_location("umcp_test_sanity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load test_sanity.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_derive_api_base_from_mcp_url(sanity_module):
    fn = sanity_module._derive_api_base_from_mcp_url
    cases = [
        ("http://localhost:8013/mcp", "http://localhost:8013/api"),
        ("http://localhost:8013/mcp/", "http://localhost:8013/api"),
        ("http://localhost:8013/x/y/mcp", "http://localhost:8013/x/y/api"),
        ("http://localhost:8013", "http://localhost:8013/api"),
        ("http://localhost:8013/", "http://localhost:8013/api"),
        ("http://localhost:8013/custom", "http://localhost:8013/custom/api"),
    ]
    for src, exp in cases:
        assert fn(src) == exp


def test_pretty_handles_unserializable_objects(sanity_module):
    pretty = sanity_module.pretty

    class NonSerializable:
        def __init__(self):
            self.data = {"a": 1}
            self.raw = b"bytes"

    text = pretty(NonSerializable())
    # Should be valid JSON string and contain our keys/values rendered safely
    assert "a" in text
    assert "bytes" in text


def test__to_plain_dataclass_and_bytes(sanity_module):
    to_plain = sanity_module._to_plain

    @dataclass
    class D:
        x: int
        y: bytes

    obj = D(3, b"hello")
    plain = to_plain(obj)
    assert isinstance(plain, dict)
    assert plain["x"] == 3
    assert plain["y"] == "hello"


def test__extract_from_content_variations(sanity_module):
    extr = sanity_module._extract_from_content

    # JSON typed content
    content = [{"type": "json", "data": {"k": 1}}]
    assert extr(content) == {"k": 1}

    # JSON embedded in text
    content = [{"type": "text", "text": "{\"a\": 2}"}]
    assert extr(content) == {"a": 2}

    # Plain text fallback
    content = [{"type": "text", "text": "hello"}]
    assert extr(content) == "hello"


def test__normalise_tool_result_shapes(sanity_module):
    norm = sanity_module._normalise_tool_result

    # Shape with content array and data
    result = {"content": [{"type": "json", "data": {"ok": True}}]}
    plain, payload = norm(result)
    assert payload == {"ok": True}

    # Shape with top-level data
    result = {"data": {"v": 5}}
    plain, payload = norm(result)
    assert payload == {"v": 5}

    # Shape with result key
    result = {"result": [1, 2, 3]}
    plain, payload = norm(result)
    assert payload == [1, 2, 3]


@pytest.mark.asyncio
async def test_ahttp_get_json_wraps_sync(sanity_module, monkeypatch):
    # Monkeypatch the sync function to avoid real HTTP
    called = {"count": 0}

    def fake_http(url: str, timeout: int = 30) -> Tuple[int, Any, dict]:
        called["count"] += 1
        return 200, {"ok": True, "url": url}, {"content-type": "application/json"}

    monkeypatch.setattr(sanity_module, "http_get_json", fake_http)

    status, body, headers = await sanity_module.ahttp_get_json("http://example/api")
    assert status == 200
    assert body["ok"] is True
    assert called["count"] == 1
