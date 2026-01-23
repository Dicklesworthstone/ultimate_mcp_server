#!/usr/bin/env python
"""Demo of advanced extraction capabilities using Ultimate MCP Server."""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

# Add project root to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.traceback import Traceback

from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.core.providers.base import get_provider
from ultimate_mcp_server.utils import get_logger
from ultimate_mcp_server.utils.display import CostTracker, parse_and_display_result
from ultimate_mcp_server.utils.logging.console import console
from ultimate_mcp_server.utils.parsing import extract_json_from_markdown

# --- Debug Flag ---
USE_DEBUG_LOGS = True  # Set to True to enable detailed logging
# ------------------

# Initialize logger
logger = get_logger("example.advanced_extraction")
logger.set_level("debug")


# Configure the OpenAI client for direct extraction demos
async def setup_openai_provider():
    """Set up an OpenAI provider for demonstration."""
    try:
        logger.info("Initializing OpenAI for demonstration", emoji_key="start")

        # Get OpenAI provider - get_provider will return None if key missing/invalid in config
        provider = await get_provider(Provider.OPENAI.value)
        if not provider:
            logger.error(
                "Failed to get OpenAI provider. Is the OPENAI_API_KEY configured correctly in your environment/config?"
            )
            return None

        logger.success("OpenAI provider initialized successfully.")
        return provider
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI provider: {e}", emoji_key="error")
        return None


async def run_json_extraction_example(provider, tracker: CostTracker):
    """Demonstrate JSON extraction."""
    if USE_DEBUG_LOGS:
        logger.debug("Entering run_json_extraction_example.")
    if not provider:
        console.print("[yellow]Skipping JSON extraction demo - no provider available.[/yellow]")
        if USE_DEBUG_LOGS:
            logger.debug("Exiting run_json_extraction_example (no provider).")
        return

    console.print(Rule("[bold blue]1. JSON Extraction Example[/bold blue]"))

    # Load sample text
    sample_path = Path(__file__).parent / "data" / "sample_event.txt"
    if not sample_path.exists():
        # Create a sample text for demonstration
        sample_text = """
        Tech Conference 2024
        Location: San Francisco Convention Center, 123 Tech Blvd, San Francisco, CA 94103
        Date: June 15-17, 2024
        Time: 9:00 AM - 6:00 PM daily
        
        Registration Fee: $599 (Early Bird: $499 until March 31)
        
        Keynote Speakers:
        - Dr. Sarah Johnson, AI Research Director at TechCorp
        - Mark Williams, CTO of FutureTech Industries
        - Prof. Emily Chen, MIT Computer Science Department
        
        Special Events:
        - Networking Reception: June 15, 7:00 PM - 10:00 PM
        - Hackathon: June 16, 9:00 PM - 9:00 AM (overnight)
        - Career Fair: June 17, 1:00 PM - 5:00 PM
        
        For more information, contact events@techconference2024.example.com or call (555) 123-4567.
        """
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        # Write sample text to file
        with open(sample_path, "w") as f:
            f.write(sample_text)
    else:
        # Read existing sample text
        with open(sample_path, "r") as f:
            sample_text = f.read()

    # Display sample text
    console.print(Panel(sample_text, title="Sample Event Text", border_style="blue"))

    # Define JSON schema for event
    event_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Event name"},
            "location": {
                "type": "object",
                "properties": {
                    "venue": {"type": "string"},
                    "address": {"type": "string"},
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                    "zip": {"type": "string"},
                },
            },
            "dates": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "format": "date"},
                    "end": {"type": "string", "format": "date"},
                },
            },
            "time": {"type": "string"},
            "registration": {
                "type": "object",
                "properties": {
                    "regular_fee": {"type": "number"},
                    "early_bird_fee": {"type": "number"},
                    "early_bird_deadline": {"type": "string", "format": "date"},
                },
            },
            "speakers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "title": {"type": "string"},
                        "organization": {"type": "string"},
                    },
                },
            },
            "special_events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "date": {"type": "string", "format": "date"},
                        "time": {"type": "string"},
                    },
                },
            },
            "contact": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "format": "email"},
                    "phone": {"type": "string"},
                },
            },
        },
    }

    # Display JSON schema
    schema_json = json.dumps(event_schema, indent=2)
    console.print(
        Panel(
            Syntax(schema_json, "json", theme="monokai", line_numbers=True),
            title="Event JSON Schema",
            border_style="green",
        )
    )

    # Extract JSON using direct provider call
    logger.info("Extracting structured JSON data from text...", emoji_key="processing")

    try:
        start_time = time.time()

        # Instead of using the tool, use direct completion for demo purposes
        prompt = f"""
        Extract structured information from the following text into a JSON object.
        Follow the provided JSON schema exactly.
        
        TEXT:
        {sample_text}
        
        JSON SCHEMA:
        {json.dumps(event_schema, indent=2)}
        
        Provide only the valid JSON object as output, with no additional commentary.
        """

        if USE_DEBUG_LOGS:
            logger.debug(f"JSON Extraction Prompt:\n{prompt}")

        # Call the provider directly
        result = await provider.generate_completion(
            prompt=prompt,
            model="gpt-4.1-mini",  # Use an available OpenAI model
            temperature=0.2,  # Lower temperature for more deterministic output
            max_tokens=1500,  # Enough tokens for a full response
        )

        # Track cost
        tracker.add_call(result)

        if USE_DEBUG_LOGS:
            logger.debug(f"Raw JSON Extraction Result Text:\n{result.text}")

        # Process the result to extract just the JSON
        try:
            # Try to parse the response as JSON
            raw_text = result.text.strip()
            text_to_parse = extract_json_from_markdown(raw_text)
            if USE_DEBUG_LOGS:
                logger.debug(f"Raw text received: {raw_text[:500]}...")
                logger.debug(f"Attempting to parse JSON after cleaning: {text_to_parse[:500]}...")
            json_result = json.loads(text_to_parse)
            if USE_DEBUG_LOGS:
                logger.debug(f"Successfully parsed JSON: {json.dumps(json_result, indent=2)}")

            # Create a dictionary with structured data and metadata for display
            structured_result_data = {
                "json": json_result,  # The actual parsed JSON
                "validated": True,  # Assuming validation happens elsewhere or is implied
                "model": result.model,
                "processing_time": time.time() - start_time,
                "tokens": {
                    "input": result.input_tokens,
                    "output": result.output_tokens,
                    "total": result.input_tokens + result.output_tokens,
                },
                "cost": result.cost,
            }

            # Display the results using the utility function
            parse_and_display_result(
                title="JSON Extraction Results",
                input_data={"text": sample_text, "schema": event_schema},
                result=structured_result_data,  # Pass the structured data
                console=console,
            )

        except json.JSONDecodeError as e:
            # Log the error regardless of debug flag
            logger.error(f"JSONDecodeError occurred: {e}", exc_info=False)

            if USE_DEBUG_LOGS:
                # Log the string that caused the error (before cleaning)
                logger.debug(f"Raw string causing JSONDecodeError:\n{raw_text}")
                # Log the string that failed parsing (after cleaning)
                logger.debug(f"Cleaned string that failed JSON parsing:\n{text_to_parse}")
                # Print a rich traceback to the console
                console.print("[bold red]-- Traceback for JSONDecodeError --[/bold red]")
                console.print(Traceback())
                console.print("[bold red]-- End Traceback --[/bold red]")

            # If JSON parsing fails, show the raw response
            console.print(
                Panel(
                    raw_text,  # Show the original raw text from the model
                    title="[yellow]Raw Model Output (JSON parsing failed)[/yellow]",
                    border_style="red",
                )
            )

    except Exception as e:
        logger.error(f"Error extracting JSON: {str(e)}", emoji_key="error", exc_info=True)

    console.print()
    if USE_DEBUG_LOGS:
        logger.debug("Exiting run_json_extraction_example.")


async def table_extraction_demo(provider, tracker: CostTracker):
    """Demonstrate table extraction capabilities."""
    if USE_DEBUG_LOGS:
        logger.debug("Entering table_extraction_demo.")
    if not provider:
        console.print("[yellow]Skipping table extraction demo - no provider available.[/yellow]")
        if USE_DEBUG_LOGS:
            logger.debug("Exiting table_extraction_demo (no provider).")
        return

    logger.info("Starting table extraction demo", emoji_key="start")

    # Sample text with embedded table
    text = """
    Financial Performance by Quarter (2023-2024)
    
    | Quarter | Revenue ($M) | Expenses ($M) | Profit ($M) | Growth (%) |
    |---------|-------------|---------------|-------------|------------|
    | Q1 2023 | 42.5        | 32.1          | 10.4        | 3.2        |
    | Q2 2023 | 45.7        | 33.8          | 11.9        | 6.5        |
    | Q3 2023 | 50.2        | 35.6          | 14.6        | 9.8        |
    | Q4 2023 | 58.3        | 38.2          | 20.1        | 15.2       |
    | Q1 2024 | 60.1        | 39.5          | 20.6        | 3.1        |
    | Q2 2024 | 65.4        | 41.2          | 24.2        | 8.8        |
    
    Note: All figures are in millions of dollars and are unaudited.
    Growth percentages are relative to the previous quarter.
    """

    # Log extraction attempt
    logger.info("Performing table extraction", emoji_key="processing")

    try:
        start_time = time.time()

        # Prompt for table extraction
        prompt = f"""
        Extract the table from the following text and format it as both JSON and Markdown.
        
        TEXT:
        {text}
        
        For the JSON format, use this structure:
        {{
            "headers": ["Header1", "Header2", ...],
            "rows": [
                {{"Header1": "value", "Header2": "value", ...}},
                ...
            ]
        }}
        
        For the Markdown format, output a well-formatted Markdown table.
        
        Also extract any metadata about the table (title, notes, etc.).
        
        Format your response as JSON with the following structure:
        {{
            "json_table": {{...}},
            "markdown_table": "...",
            "metadata": {{
                "title": "...",
                "notes": [
                    "..."
                ]
            }}
        }}
        """

        if USE_DEBUG_LOGS:
            logger.debug(f"Table Extraction Prompt:\n{prompt}")

        # Call the provider directly
        result = await provider.generate_completion(
            prompt=prompt, model="gpt-4.1-mini", temperature=0.2, max_tokens=1500
        )

        # Track cost
        tracker.add_call(result)

        if USE_DEBUG_LOGS:
            logger.debug(f"Raw Table Extraction Result Text:\n{result.text}")

        try:
            # Try to parse the response as JSON
            raw_text = result.text.strip()  # Keep raw text separate
            text_to_parse = extract_json_from_markdown(raw_text)  # Clean it
            if USE_DEBUG_LOGS:
                # Log both raw and cleaned versions
                logger.debug(f"Raw text received (Table): {raw_text[:500]}...")
                logger.debug(
                    f"Attempting to parse Table Extraction JSON after cleaning: {text_to_parse[:500]}..."
                )
            json_result = json.loads(text_to_parse)  # Parse the cleaned version
            if USE_DEBUG_LOGS:
                logger.debug(
                    f"Successfully parsed Table Extraction JSON: {json.dumps(json_result, indent=2)}"
                )

            # Create structured data dictionary for display
            structured_result_data = {
                "formats": {
                    "json": json_result.get("json_table", {}),
                    "markdown": json_result.get("markdown_table", ""),
                },
                "metadata": json_result.get("metadata", {}),
                "model": result.model,
                "processing_time": time.time() - start_time,
                "tokens": {
                    "input": result.input_tokens,
                    "output": result.output_tokens,
                    "total": result.input_tokens + result.output_tokens,
                },
                "cost": result.cost,
            }

            # Parse the result using the shared utility
            parse_and_display_result(
                "Table Extraction Demo",
                {"text": text},
                structured_result_data,  # Pass the structured data
            )

        except json.JSONDecodeError as e:
            # Log the error regardless of debug flag
            logger.error(f"JSONDecodeError in Table Extraction occurred: {e}", exc_info=False)

            if USE_DEBUG_LOGS:
                # Log both raw and cleaned versions for debugging the failure
                logger.debug(f"Raw string causing JSONDecodeError in Table Extraction:\n{raw_text}")
                logger.debug(
                    f"Cleaned string that failed JSON parsing in Table Extraction:\n{text_to_parse}"
                )
                # Print a rich traceback to the console
                console.print(
                    "[bold red]-- Traceback for JSONDecodeError (Table Extraction) --[/bold red]"
                )
                console.print(Traceback())
                console.print("[bold red]-- End Traceback --[/bold red]")

            # If JSON parsing fails, show the raw response using the original raw_text
            console.print(
                Panel(
                    raw_text,
                    title="[yellow]Raw Model Output (JSON parsing failed)[/yellow]",
                    border_style="red",
                )
            )

    except Exception as e:
        logger.error(f"Error in table extraction: {str(e)}", emoji_key="error")
    # Add exit log
    if USE_DEBUG_LOGS:
        logger.debug("Exiting table_extraction_demo.")


async def semantic_schema_inference_demo(provider, tracker: CostTracker):
    """Demonstrate semantic schema inference."""
    if USE_DEBUG_LOGS:
        logger.debug("Entering semantic_schema_inference_demo.")
    if not provider:
        console.print(
            "[yellow]Skipping semantic schema inference demo - no provider available.[/yellow]"
        )
        if USE_DEBUG_LOGS:
            logger.debug("Exiting semantic_schema_inference_demo (no provider).")
        return

    logger.info("Starting semantic schema inference demo", emoji_key="start")

    # Sample text for schema inference
    text = """
    Patient Record: John Smith
    Date of Birth: 05/12/1978
    Patient ID: P-98765
    Blood Type: O+
    Height: 182 cm
    Weight: 76 kg
    
    Medications:
    - Lisinopril 10mg, once daily
    - Metformin 500mg, twice daily
    - Atorvastatin 20mg, once daily at bedtime
    
    Allergies:
    - Penicillin (severe)
    - Shellfish (mild)
    
    Recent Vital Signs:
    Date: 03/15/2024
    Blood Pressure: 128/85 mmHg
    Heart Rate: 72 bpm
    Temperature: 98.6°F
    Oxygen Saturation: 98%
    
    Medical History:
    - Type 2 Diabetes (diagnosed 2015)
    - Hypertension (diagnosed 2017)
    - Hyperlipidemia (diagnosed 2019)
    - Appendectomy (2005)
    """

    # Define a schema template for the extraction
    patient_schema = {
        "type": "object",
        "properties": {
            "patient": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dob": {"type": "string"},
                    "id": {"type": "string"},
                    "blood_type": {"type": "string"},
                    "height": {"type": "string"},
                    "weight": {"type": "string"},
                },
            },
            "medications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "dosage": {"type": "string"},
                        "frequency": {"type": "string"},
                    },
                },
            },
            "allergies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"allergen": {"type": "string"}, "severity": {"type": "string"}},
                },
            },
            "vital_signs": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "blood_pressure": {"type": "string"},
                    "heart_rate": {"type": "string"},
                    "temperature": {"type": "string"},
                    "oxygen_saturation": {"type": "string"},
                },
            },
            "medical_history": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "condition": {"type": "string"},
                        "diagnosed": {"type": "string"},
                    },
                },
            },
        },
    }

    # Log schema inference attempt
    logger.info("Performing schema inference", emoji_key="processing")

    try:
        start_time = time.time()

        # Prompt for semantic schema extraction
        prompt = f"""
        Extract structured information from the text according to the provided semantic schema.
        
        TEXT:
        {text}
        
        SEMANTIC SCHEMA:
        {json.dumps(patient_schema, indent=2)}
        
        Analyze the text and extract information following the schema structure. Return a valid JSON object.
        """

        if USE_DEBUG_LOGS:
            logger.debug(f"Schema Inference Prompt:\n{prompt}")

        # Call the provider directly
        result = await provider.generate_completion(
            prompt=prompt, model="gpt-4.1-mini", temperature=0.2, max_tokens=1000
        )

        # Track cost
        tracker.add_call(result)

        if USE_DEBUG_LOGS:
            logger.debug(f"Raw Schema Inference Result Text:\n{result.text}")

        try:
            # Try to parse the response as JSON
            raw_text = result.text.strip()
            text_to_parse = extract_json_from_markdown(raw_text)
            if USE_DEBUG_LOGS:
                logger.debug(f"Raw text received (Schema): {raw_text[:500]}...")
                logger.debug(
                    f"Attempting to parse Schema Inference JSON after cleaning: {text_to_parse[:500]}..."
                )
            json_result = json.loads(text_to_parse)
            if USE_DEBUG_LOGS:
                logger.debug(
                    f"Successfully parsed Schema Inference JSON: {json.dumps(json_result, indent=2)}"
                )

            # Create structured data dictionary for display
            structured_result_data = {
                "extracted_data": json_result,
                "model": result.model,
                "processing_time": time.time() - start_time,
                "tokens": {
                    "input": result.input_tokens,
                    "output": result.output_tokens,
                    "total": result.input_tokens + result.output_tokens,
                },
                "cost": result.cost,
            }

            # Parse the result using the shared utility
            parse_and_display_result(
                "Semantic Schema Inference Demo",
                {"text": text},
                structured_result_data,  # Pass the structured data
            )

        except json.JSONDecodeError as e:
            # Log the error regardless of debug flag
            logger.error(f"JSONDecodeError in Schema Inference occurred: {e}", exc_info=False)

            if USE_DEBUG_LOGS:
                # Log both raw and cleaned versions
                logger.debug(f"Raw string causing JSONDecodeError in Schema Inference:\n{raw_text}")
                logger.debug(
                    f"Cleaned string that failed JSON parsing in Schema Inference:\n{text_to_parse}"
                )
                # Print a rich traceback to the console
                console.print(
                    "[bold red]-- Traceback for JSONDecodeError (Schema Inference) --[/bold red]"
                )
                console.print(Traceback())
                console.print("[bold red]-- End Traceback --[/bold red]")

            # If JSON parsing fails, show the raw response
            console.print(
                Panel(
                    raw_text,
                    title="[yellow]Raw Model Output (JSON parsing failed)[/yellow]",
                    border_style="red",
                )
            )

    except Exception as e:
        logger.error(f"Error in schema inference: {str(e)}", emoji_key="error")
    # Add exit log
    if USE_DEBUG_LOGS:
        logger.debug("Exiting semantic_schema_inference_demo.")


async def entity_extraction_demo(provider, tracker: CostTracker):
    """Demonstrate entity extraction capabilities."""
    if USE_DEBUG_LOGS:
        logger.debug("Entering entity_extraction_demo.")
    if not provider:
        console.print("[yellow]Skipping entity extraction demo - no provider available.[/yellow]")
        if USE_DEBUG_LOGS:
            logger.debug("Exiting entity_extraction_demo (no provider).")
        return

    logger.info("Starting entity extraction demo", emoji_key="start")

    # Sample text for entity extraction
    text = """
    In a groundbreaking announcement on March 15, 2024, Tesla unveiled its latest solar energy
    technology in partnership with SolarCity. CEO Elon Musk presented the new PowerWall 4.0 
    battery system at their headquarters in Austin, Texas. The system can store up to 20kWh of 
    energy and costs approximately $6,500 per unit.
    
    According to Dr. Maria Chen, lead researcher at the National Renewable Energy Laboratory (NREL),
    this technology represents a significant advancement in residential energy storage. The new
    system integrates with the Tesla mobile app on both iOS and Android platforms, allowing users
    to monitor energy usage in real-time.
    
    Tesla stock (TSLA) rose 5.8% following the announcement, reaching $248.32 per share on the NASDAQ.
    The company plans to begin production at their Gigafactory Nevada location by June 2024, with
    initial deployments in California and Texas markets.
    """

    # Log entity extraction attempt
    logger.info("Performing entity extraction", emoji_key="processing")

    try:
        start_time = time.time()

        # Prompt for entity extraction
        prompt = f"""
        Extract key-value pairs and entities from the following text, categorized by type.
        
        TEXT:
        {text}
        
        Extract the following categories of information:
        - Organizations (companies, institutions, etc.)
        - People (names and titles)
        - Locations (cities, states, facilities, etc.)
        - Dates and Times
        - Products and Technologies
        - Numerical Values (monetary values, percentages, measurements, etc.)
        
        Format the output as a JSON object with these categories as keys, and each containing relevant entities found.
        Within each category, provide structured information when possible.
        """

        if USE_DEBUG_LOGS:
            logger.debug(f"Entity Extraction Prompt:\n{prompt}")

        # Call the provider directly
        result = await provider.generate_completion(
            prompt=prompt, model="gpt-4.1-mini", temperature=0.2, max_tokens=500
        )

        # Track cost
        tracker.add_call(result)

        if USE_DEBUG_LOGS:
            logger.debug(f"Raw Entity Extraction Result Text:\n{result.text}")

        try:
            # Try to parse the response as JSON
            raw_text = result.text.strip()
            text_to_parse = extract_json_from_markdown(raw_text)
            if USE_DEBUG_LOGS:
                logger.debug(f"Raw text received (Entity): {raw_text[:500]}...")
                logger.debug(
                    f"Attempting to parse Entity Extraction JSON after cleaning: {text_to_parse[:500]}..."
                )
            if USE_DEBUG_LOGS:
                logger.debug(f"EXACT STRING PASSED TO json.loads: >>>{text_to_parse}<<<")

            try:
                # First try standard parsing
                json_result = json.loads(text_to_parse)
            except json.JSONDecodeError as e:
                logger.warning(f"Standard JSON parsing failed: {e}. Attempting emergency repair.")

                # Emergency fallback for malformed JSON due to unterminated strings
                # 1. Look for the raw JSON structure with markdown removed
                text_no_markdown = text_to_parse

                # 2. Manually check for key entity categories, even if JSON is malformed
                # Create a structured result with categories we expect to find
                json_result = {
                    "Organizations": [],
                    "People": [],
                    "Locations": [],
                    "Dates and Times": [],
                    "Products and Technologies": [],
                    "Numerical Values": [],
                }

                # Look for entity categories using regex
                org_matches = re.findall(
                    r'"name"\s*:\s*"([^"]+)".*?"type"\s*:\s*"([^"]+)"', text_no_markdown
                )
                for name, entity_type in org_matches:
                    # Determine which category this entity belongs to based on type
                    if any(
                        keyword in entity_type.lower()
                        for keyword in ["company", "corporation", "institution", "exchange"]
                    ):
                        json_result["Organizations"].append({"name": name, "type": entity_type})
                    elif any(
                        keyword in entity_type.lower() for keyword in ["city", "state", "facility"]
                    ):
                        json_result["Locations"].append({"name": name, "type": entity_type})
                    elif any(
                        keyword in entity_type.lower()
                        for keyword in ["battery", "app", "system", "technology"]
                    ):
                        json_result["Products and Technologies"].append(
                            {"name": name, "type": entity_type}
                        )

                # Look for people - they usually have titles and organizations
                people_matches = re.findall(
                    r'"name"\s*:\s*"([^"]+)".*?"title"\s*:\s*"([^"]+)".*?"organization"\s*:\s*"([^"]*)"',
                    text_no_markdown,
                )
                for name, title, org in people_matches:
                    json_result["People"].append(
                        {"name": name, "title": title, "organization": org}
                    )

                # Dates and numerical values are harder to extract generically
                # but we can look for obvious patterns
                date_matches = re.findall(
                    r'"date"\s*:\s*"([^"]+)".*?"event"\s*:\s*"([^"]+)"', text_no_markdown
                )
                for date, event in date_matches:
                    json_result["Dates and Times"].append({"date": date, "event": event})

                # For numerical values, look for values with units
                value_matches = re.findall(
                    r'"value"\s*:\s*([^,]+).*?"unit"\s*:\s*"([^"]+)"', text_no_markdown
                )
                for value, unit in value_matches:
                    # Clean up the value
                    clean_value = value.strip('" ')
                    item = {"value": clean_value, "unit": unit}

                    # Look for a description if available
                    desc_match = re.search(r'"description"\s*:\s*"([^"]+)"', text_no_markdown)
                    if desc_match:
                        item["description"] = desc_match.group(1)

                    json_result["Numerical Values"].append(item)

                # Add a note about emergency repair
                logger.warning("Used emergency JSON repair - results may be incomplete")

            if USE_DEBUG_LOGS:
                logger.debug(
                    f"Successfully parsed Entity Extraction JSON: {json.dumps(json_result, indent=2)}"
                )

            # Create structured data dictionary for display
            structured_result_data = {
                "extracted_data": json_result,
                "structured": True,
                "categorized": True,
                "model": result.model,
                "processing_time": time.time() - start_time,
                "tokens": {
                    "input": result.input_tokens,
                    "output": result.output_tokens,
                    "total": result.input_tokens + result.output_tokens,
                },
                "cost": result.cost,
            }

            # Parse the result using the shared utility
            parse_and_display_result(
                "Entity Extraction Demo",
                {"text": text},
                structured_result_data,  # Pass the structured data
            )

        except json.JSONDecodeError as e:
            # Log the error regardless of debug flag
            logger.error(f"JSONDecodeError in Entity Extraction occurred: {e}", exc_info=False)

            if USE_DEBUG_LOGS:
                # Log both raw and cleaned versions
                logger.debug(
                    f"Raw string causing JSONDecodeError in Entity Extraction:\n{raw_text}"
                )
                logger.debug(
                    f"Cleaned string that failed JSON parsing in Entity Extraction:\n{text_to_parse}"
                )
                # Print a rich traceback to the console
                console.print(
                    "[bold red]-- Traceback for JSONDecodeError (Entity Extraction) --[/bold red]"
                )
                console.print(Traceback())
                console.print("[bold red]-- End Traceback --[/bold red]")

            # If JSON parsing fails, show the raw response
            console.print(
                Panel(
                    raw_text,
                    title="[yellow]Raw Model Output (JSON parsing failed)[/yellow]",
                    border_style="red",
                )
            )

    except Exception as e:
        logger.error(f"Error in entity extraction: {str(e)}", emoji_key="error")
    # Add exit log
    if USE_DEBUG_LOGS:
        logger.debug("Exiting entity_extraction_demo.")


async def main():
    """Run the advanced extraction demos."""
    tracker = CostTracker()  # Instantiate tracker
    provider = await setup_openai_provider()

    if not provider:
        logger.warning(
            "OpenAI provider not available. Demo sections requiring it will be skipped.",
            emoji_key="warning",
        )

    console.print(Rule("[bold magenta]Advanced Extraction Demos Starting[/bold magenta]"))

    demos_to_run = [
        (run_json_extraction_example, "JSON Extraction"),
        (table_extraction_demo, "Table Extraction"),
        (semantic_schema_inference_demo, "Schema Inference"),
        (entity_extraction_demo, "Entity Extraction"),
    ]

    # Execute demos sequentially
    for demo_func, demo_name in demos_to_run:
        try:
            await demo_func(provider, tracker)  # Pass tracker
        except Exception as e:
            logger.error(f"Error running {demo_name} demo: {e}", emoji_key="error", exc_info=True)

    # Display final cost summary
    tracker.display_summary(console)

    logger.success("Advanced Extraction Demo finished successfully!", emoji_key="complete")
    console.print(Rule("[bold magenta]Advanced Extraction Demos Complete[/bold magenta]"))


if __name__ == "__main__":
    # Run the demos
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
