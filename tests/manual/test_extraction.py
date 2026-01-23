#!/usr/bin/env python3
"""
Manual test for extraction tools using standardized completion.
This script tests the key functions in extraction.py to ensure they work
with the updated standardized completion tool.
"""

import asyncio
import json
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.tools.extraction import (
    extract_json,
    extract_key_value_pairs,
    extract_table,
)


async def test_extract_json():
    """Test the extract_json function with a simple JSON object."""
    print("\n--- Testing extract_json ---")
    # Simplified JSON without nested structures
    sample_text = """
    Here's the result of my analysis:
    
    {
      "name": "John Smith",
      "age": 42,
      "skills": "programming, design, project management",
      "email": "john@example.com",
      "phone": "555-1234"
    }
    
    Let me know if you need any more information.
    """

    result = await extract_json(
        text=sample_text, provider=Provider.OPENAI.value, model="gpt-3.5-turbo"
    )

    print(f"Success: {result.get('success', False)}")
    print(f"Model used: {result.get('model', 'unknown')}")
    print(f"Tokens: {result.get('tokens', {})}")
    print(f"Processing time: {result.get('processing_time', 0):.2f}s")

    # Pretty print the extracted data
    if result.get("data"):
        print("Extracted JSON:")
        print(json.dumps(result["data"], indent=2))
    else:
        print("Failed to extract JSON")
        print(f"Error: {result.get('error', 'unknown error')}")


async def test_extract_table():
    """Test the extract_table function with a simple table."""
    print("\n--- Testing extract_table ---")
    sample_text = """
    Here's a summary of our quarterly sales:
    
    | Product  | Q1 Sales | Q2 Sales |
    |----------|----------|----------|
    | Widget A | 1200     | 1350     |
    | Widget B | 850      | 940      |
    
    As you can see, Widget A performed best in Q2.
    """

    result = await extract_table(
        text=sample_text,
        return_formats=["json"],  # Just request json to keep it simple
        provider=Provider.OPENAI.value,
        model="gpt-3.5-turbo",
    )

    print(f"Success: {result.get('success', False)}")
    print(f"Model used: {result.get('model', 'unknown')}")
    print(f"Tokens: {result.get('tokens', {})}")
    print(f"Processing time: {result.get('processing_time', 0):.2f}s")

    # Print the extracted data
    if result.get("data"):
        print("Extracted Table Data:")
        if isinstance(result["data"], dict) and "json" in result["data"]:
            print("JSON Format:")
            print(json.dumps(result["data"]["json"], indent=2))
        else:
            print(json.dumps(result["data"], indent=2))
    else:
        print("Failed to extract table")
        print(f"Error: {result.get('error', 'unknown error')}")
        if result.get("raw_text"):
            print(f"Raw text: {result.get('raw_text')[:200]}...")


async def test_extract_key_value_pairs():
    """Test the extract_key_value_pairs function."""
    print("\n--- Testing extract_key_value_pairs ---")
    sample_text = """
    Patient Information:
    
    Name: Jane Doe
    DOB: 05/12/1985
    Gender: Female
    Blood Type: O+
    Height: 5'6"
    Weight: 145 lbs
    Allergies: Penicillin, Shellfish
    Primary Care Physician: Dr. Robert Chen
    """

    result = await extract_key_value_pairs(
        text=sample_text, provider=Provider.OPENAI.value, model="gpt-3.5-turbo"
    )

    print(f"Success: {result.get('success', False)}")
    print(f"Model used: {result.get('model', 'unknown')}")
    print(f"Tokens: {result.get('tokens', {})}")
    print(f"Processing time: {result.get('processing_time', 0):.2f}s")

    # Print the extracted data
    if result.get("data"):
        print("Extracted Key-Value Pairs:")
        for key, value in result["data"].items():
            print(f"  {key}: {value}")
    else:
        print("Failed to extract key-value pairs")
        print(f"Error: {result.get('error', 'unknown error')}")


async def main():
    """Run all tests."""
    print("Testing extraction tools with standardized completion...")

    await test_extract_json()
    await test_extract_table()
    await test_extract_key_value_pairs()

    print("\nAll tests completed.")


if __name__ == "__main__":
    asyncio.run(main())
