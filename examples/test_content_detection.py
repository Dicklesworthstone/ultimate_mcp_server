#!/usr/bin/env python
"""
Test script to demonstrate enhanced content type detection with Magika integration
in the DocumentProcessingTool.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports when running as script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from ultimate_mcp_server.core.server import Gateway  # noqa: E402
from ultimate_mcp_server.tools.document_conversion_and_processing import (  # noqa: E402
    DocumentProcessingTool,  # noqa: E402
)

console = Console()

# Sample content for testing
HTML_CONTENT = """<!DOCTYPE html>
<html>
<head>
    <title>Test HTML Document</title>
    <meta charset="utf-8">
</head>
<body>
    <h1>This is a test HTML document</h1>
    <p>This paragraph is for testing the content detection.</p>
    <div class="container">
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
    </div>
    <script>
        // Some JavaScript
        console.log("Hello world");
    </script>
</body>
</html>
"""

MARKDOWN_CONTENT = """# Test Markdown Document

This is a paragraph in markdown format.

## Section 1

* Item 1
* Item 2

[Link to example](https://example.com)

```python
def hello_world():
    print("Hello world")
```

| Column 1 | Column 2 |
|----------|----------|
| Cell 1   | Cell 2   |
"""

CODE_CONTENT = """
#!/usr/bin/env python
import sys
from typing import List, Dict, Optional

class TestClass:
    def __init__(self, name: str, value: int = 0):
        self.name = name
        self.value = value
        
    def process(self, data: List[Dict]) -> Optional[Dict]:
        result = {}
        for item in data:
            if "key" in item:
                result[item["key"]] = item["value"]
        return result if result else None
        
def main():
    test = TestClass("test", 42)
    result = test.process([{"key": "a", "value": 1}, {"key": "b", "value": 2}])
    print(f"Result: {result}")
    
if __name__ == "__main__":
    main()
"""

PLAIN_TEXT_CONTENT = """
This is a plain text document with no special formatting.

It contains multiple paragraphs and some sentences.
There are no markdown elements, HTML tags, or code structures.

Just regular text that someone might write in a simple text editor.
"""

AMBIGUOUS_CONTENT = """
Here's some text with a <div> tag in it.

# This looks like a heading

But it also has some <span>HTML</span> elements.

def is_this_code():
    return "maybe"

Regular paragraph text continues here.
"""


async def test_content_detection():
    console.print(
        Panel("Testing Content Type Detection with Magika Integration", style="bold green")
    )

    # Initialize the document processor
    gateway = Gateway("content-detection-test")
    # Initialize providers
    console.print("Initializing gateway and providers...")
    await gateway._initialize_providers()

    # Create document processing tool
    doc_tool = DocumentProcessingTool(gateway)

    # Define test cases
    test_cases = [
        ("HTML Document", HTML_CONTENT),
        ("Markdown Document", MARKDOWN_CONTENT),
        ("Code Document", CODE_CONTENT),
        ("Plain Text Document", PLAIN_TEXT_CONTENT),
        ("Ambiguous Content", AMBIGUOUS_CONTENT),
    ]

    # Create results table
    results_table = Table(title="Content Type Detection Results")
    results_table.add_column("Content Type", style="cyan")
    results_table.add_column("Detected Type", style="green")
    results_table.add_column("Confidence", style="yellow")
    results_table.add_column("Method", style="magenta")
    results_table.add_column("Detection Criteria", style="blue")

    # Test each case
    for name, content in test_cases:
        console.print(f"\nDetecting content type for: [bold cyan]{name}[/]")

        # Detect content type
        result = await doc_tool.detect_content_type(content)

        # Get detection details
        detected_type = result.get("content_type", "unknown")
        confidence = result.get("confidence", 0.0)
        criteria = ", ".join(result.get("detection_criteria", []))

        # Check if Magika was used
        method = "Magika" if result.get("detection_method") == "magika" else "Heuristic"
        if not result.get("detection_method") == "magika" and result.get("magika_details"):
            method = "Combined (Magika + Heuristic)"

        # Add to results table
        results_table.add_row(
            name,
            detected_type,
            f"{confidence:.2f}",
            method,
            criteria[:100] + "..." if len(criteria) > 100 else criteria,
        )

        # Show all scores
        scores = result.get("all_scores", {})
        if scores:
            scores_table = Table(title="Detection Scores")
            scores_table.add_column("Content Type", style="cyan")
            scores_table.add_column("Score", style="yellow")

            for ctype, score in scores.items():
                scores_table.add_row(ctype, f"{score:.3f}")

            console.print(scores_table)

        # Show Magika details if available
        if "magika_details" in result:
            magika_details = result["magika_details"]
            console.print(
                Panel(
                    f"Magika Type: {magika_details.get('type', 'unknown')}\n"
                    f"Magika Confidence: {magika_details.get('confidence', 0.0):.3f}\n"
                    f"Matched Primary Type: {magika_details.get('matched_primary_type', False)}",
                    title="Magika Details",
                    style="blue",
                )
            )

    # Print final results table
    console.print("\n")
    console.print(results_table)

    # Now test HTML to Markdown conversion with a clearly broken HTML case
    console.print(
        Panel("Testing HTML to Markdown Conversion with Content Detection", style="bold green")
    )

    # Create a test case with problematic HTML (the one that previously failed)
    problematic_html = """<!DOCTYPE html>
<html class="client-nojs vector-feature-language-in-header-enabled vector-feature-language-in-main-page-header-disabled">
<head>
<meta charset="UTF-8">
<title>Transformer (deep learning architecture) - Wikipedia</title>
<script>(function(){var className="client-js vector-feature-language-in-header-enabled vector-feature-language-in-main-page-header-disabled";</script>
</head>
<body>
<h1>Transformer Model</h1>
<p>The Transformer is a deep learning model introduced in the paper "Attention Is All You Need".</p>
</body>
</html>"""

    console.print("Converting problematic HTML to Markdown...")
    result = await doc_tool.clean_and_format_text_as_markdown(
        text=problematic_html, extraction_method="auto", preserve_tables=True, preserve_links=True
    )

    console.print(
        Panel(
            f"Original Type: {result.get('original_content_type', 'unknown')}\n"
            f"Was HTML: {result.get('was_html', False)}\n"
            f"Extraction Method: {result.get('extraction_method_used', 'none')}",
            title="Conversion Details",
            style="cyan",
        )
    )

    console.print(
        Panel(
            result.get("markdown_text", "No markdown produced"),
            title="Converted Markdown",
            style="green",
        )
    )


if __name__ == "__main__":
    asyncio.run(test_content_detection())
