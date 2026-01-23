#!/usr/bin/env python3
"""
Manual test for advanced extraction tools using standardized completion.
This script tests the remaining extraction tools that were refactored to use
the standardized completion tool.
"""

import asyncio
import json
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.tools.extraction import extract_code_from_response, extract_semantic_schema


async def test_extract_semantic_schema():
    """Test the extract_semantic_schema function with a simple schema."""
    print("\n--- Testing extract_semantic_schema ---")

    # Define a JSON schema to extract data
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "phone": {"type": "string"},
            "interests": {"type": "array", "items": {"type": "string"}},
        },
    }

    # Sample text containing information matching the schema
    sample_text = """
    Profile information:
    
    Name: Sarah Johnson
    Contact: sarah.j@example.com
    Phone Number: 555-987-6543
    
    Sarah is interested in: machine learning, data visualization, and hiking.
    """

    result = await extract_semantic_schema(
        text=sample_text,
        semantic_schema=schema,
        provider=Provider.OPENAI.value,
        model="gpt-3.5-turbo",
    )

    print(f"Success: {result.get('success', False)}")
    print(f"Model used: {result.get('model', 'unknown')}")
    print(f"Tokens: {result.get('tokens', {})}")
    print(f"Processing time: {result.get('processing_time', 0):.2f}s")

    # Pretty print the extracted data
    if result.get("data"):
        print("Extracted Schema Data:")
        print(json.dumps(result["data"], indent=2))
    else:
        print("Failed to extract schema data")
        print(f"Error: {result.get('error', 'unknown error')}")


async def test_extract_code_from_response():
    """Test the extract_code_from_response function."""
    print("\n--- Testing extract_code_from_response ---")

    # Sample text with a code block
    sample_text = """
    Here's a Python function to calculate the factorial of a number:
    
    ```python
    def factorial(n):
        if n == 0 or n == 1:
            return 1
        else:
            return n * factorial(n-1)
            
    # Example usage
    print(factorial(5))  # Output: 120
    ```
    
    This uses a recursive approach to calculate the factorial.
    """

    # Test with regex-based extraction
    print("Testing regex-based extraction...")
    extracted_code = await extract_code_from_response(
        response_text=sample_text, model="openai/gpt-3.5-turbo", timeout=10
    )

    print("Extracted Code:")
    print(extracted_code)

    # Test with LLM-based extraction on text without markdown
    print("\nTesting LLM-based extraction...")
    sample_text_no_markdown = """
    Here's a Python function to calculate the factorial of a number:
    
    def factorial(n):
        if n == 0 or n == 1:
            return 1
        else:
            return n * factorial(n-1)
            
    # Example usage
    print(factorial(5))  # Output: 120
    
    This uses a recursive approach to calculate the factorial.
    """

    extracted_code = await extract_code_from_response(
        response_text=sample_text_no_markdown, model="openai/gpt-3.5-turbo", timeout=10
    )

    print("Extracted Code:")
    print(extracted_code)


async def main():
    """Run all tests."""
    print("Testing advanced extraction tools with standardized completion...")

    await test_extract_semantic_schema()
    await test_extract_code_from_response()

    print("\nAll tests completed.")


if __name__ == "__main__":
    asyncio.run(main())
