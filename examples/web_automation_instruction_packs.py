# examples/web_automation_instruction_packs.py

"""
Instruction Packs for the Abstract `find_and_download_pdfs` Tool.

This file contains pre-defined instruction dictionaries that configure the behavior
of the generic `find_and_download_pdfs` tool for specific tasks.
"""

# --- Instruction Pack 1: Academic Papers (arXiv) ---
ACADEMIC_PAPER_INSTRUCTIONS = {
    "search_phase": {
        "search_query_template": "site:arxiv.org {topic} filetype:pdf",
        "target_site_identification_prompt": """Analyze the search results summary for "{search_term}".
Identify the URL that is most likely a direct PDF link (ending in .pdf) or a specific relevant paper's abstract page on arXiv.org. PRIORITIZE direct PDF links over abstract pages. Always choose direct PDF links when available.

Search Results Summary:
---
{search_results_summary}
---

Respond ONLY with a valid JSON object: {{"target_url": "URL_or_null"}}.""",
        "search_engine": "google",  # Changed from duckduckgo to google which is more stable
        "num_search_results_per_query": 8,  # Increased number of results to find more PDFs
    },
    "exploration_phase": {
        "exploration_goal_prompt": """You are an AI assistant tasked with finding and downloading PDF research papers from arXiv related to '{topic}'.

Your goal is to find PDF download links for relevant papers. Look for links labeled 'PDF', 'Download PDF', or links ending in .pdf within the page.

IMPORTANT: When you find a PDF link, you MUST use the "download_pdf" action with the full PDF URL to download it. Do not try to summarize the content - downloading the PDF is the primary goal.

Please follow these guidelines:
1. If the current page is a direct PDF or has PDF in the URL, use "download_pdf" action immediately
2. If you're on an abstract page, look for PDF links and use "download_pdf" action
3. If you're on a search results page, look for relevant paper links and click them
4. Use "scroll" action to see more results if needed
5. If you can't find relevant papers after multiple steps, use "goal_impossible"
6. If you successfully download at least one PDF, use "goal_achieved"

Remember: Your PRIMARY goal is to DOWNLOAD PDFs using the "download_pdf" action, not just navigate or summarize.""",
        "navigation_keywords": [
            "abstract",
            "pdf",
            "view",
            "download",
            "related",
            "version",
            "year",
            "author",
            "subject",
            "search",
        ],
        "pdf_keywords": ["pdf", "download pdf", "download"],
        "pdf_url_patterns": [r"/pdf/\d+\.\d+v?\d*", r"\.pdf$"],  # Updated arXiv pattern
        "max_steps": 10,
        "valid_actions": [
            "click",
            "scroll",
            "download_pdf",
            "go_back",
            "goal_achieved",
            "goal_impossible",
        ],
    },
    "download_phase": {
        "metadata_extraction_prompt": """Based on the context below (URL, surrounding text/elements, often from an arXiv abstract page) for the PDF link below, extract the paper's TITLE, the primary AUTHOR's last name, and the YEAR of publication (YYYY).

Context:
---
{context}
---

Respond ONLY with a valid JSON object: {{"title": "...", "author_lastname": "...", "year": "YYYY"}}. Use "Unknown" or the current year if a value cannot be found.""",
        "filename_template": "{year}_{author_lastname}_{topic}_{title}",
        "required_metadata": ["title", "author_lastname", "year"],
    },
}

# --- Instruction Pack 2: Government Reports ---
GOVERNMENT_REPORT_INSTRUCTIONS = {
    "search_phase": {
        "search_query_template": '"{topic}" official report site:gov.uk OR site:*.gov OR site:*.gov.au OR site:*.gc.ca',  # Added more gov domains
        "target_site_identification_prompt": """Analyze the search results summary for "{search_term}".
Identify the single most promising URL pointing to an official government webpage (e.g., *.gov.uk, *.gov, *.gc.ca, *.gov.au) or official agency site likely hosting the definitive report or publication page for the topic. Avoid news articles or commentary sites.

Search Results Summary:
---
{search_results_summary}
---

Respond ONLY with a valid JSON object: {{"target_url": "URL_or_null"}}.""",
        "search_engine": "google",
    },
    "exploration_phase": {
        "exploration_goal_prompt": "Explore the official government website related to '{topic}' to find and download the primary official report(s) or policy document(s) in PDF format.",
        "navigation_keywords": [
            "publication",
            "report",
            "document",
            "research",
            "policy",
            "consultation",
            "guidance",
            "download",
            "library",
            "archive",
            "statistics",
            "data",
        ],
        "pdf_keywords": [
            "pdf",
            "download",
            "full report",
            "final report",
            "publication",
            "document",
            "read",
            "view",
            "annex",
            "appendix",
            "data",
        ],
        "pdf_url_patterns": [r"\.pdf(\?|$)", r"/assets/", r"/download/", r"/file/", r"/media/"],
        "max_steps": 15,
    },
    "download_phase": {
        "metadata_extraction_prompt": """Based on the context (URL, surrounding text/elements) for the government document PDF link below, determine the PUBLICATION_DATE (format YYYY-MM-DD, or YYYY-MM, or just YYYY if only year is available) and a concise DOCUMENT_TYPE (e.g., 'Policy Paper', 'Impact Assessment', 'Consultation Response', 'Research Report', 'Official Guidance', 'Statistics Release').

Context:
---
{context}
---

Respond ONLY with a valid JSON object: {{"date": "...", "document_type": "..."}}. Use best guess (e.g., {datetime.now().strftime('%Y-%m-%d')}, 'Report') if a value cannot be reliably determined.""",
        "filename_template": "{date}_GovReport_{topic}_{document_type}",
        "required_metadata": ["date", "document_type"],
    },
}

# --- Instruction Pack 3: Product Manuals/Datasheets ---
PRODUCT_MANUAL_INSTRUCTIONS = {
    "search_phase": {
        "search_query_template": "{topic} official manual OR datasheet OR support download PDF",  # Broadened query slightly
        "target_site_identification_prompt": """Analyze the search results summary for "{search_term}".
Identify the single URL most likely leading to the official manufacturer's product support, downloads, or manual page for the specified product. Prioritize the manufacturer's own domain. Avoid retailer sites (like Amazon, BestBuy) or general review sites.

Search Results Summary:
---
{search_results_summary}
---

Respond ONLY with a valid JSON object: {{"target_url": "URL_or_null"}}.""",
        "search_engine": "google",
    },
    "exploration_phase": {
        "exploration_goal_prompt": "Explore the manufacturer's website for the product '{topic}' to find the primary user manual, user guide, or technical datasheet available as a PDF download. Look in 'Support', 'Downloads', or 'Documentation' sections.",
        "navigation_keywords": [
            "support",
            "download",
            "manual",
            "documentation",
            "guide",
            "specification",
            "datasheet",
            "product",
            "resource",
            "driver",
            "software",
            "firmware",
        ],
        "pdf_keywords": [
            "manual",
            "guide",
            "datasheet",
            "specification",
            "pdf",
            "download",
            "instructions",
            "service manual",
            "user guide",
        ],
        "pdf_url_patterns": [
            r"manual.*\.pdf",
            r"datasheet.*\.pdf",
            r"\.pdf(\?|$)",
            r"guide.*\.pdf",
            r"spec.*\.pdf",
        ],
        "max_steps": 12,
    },
    "download_phase": {
        "metadata_extraction_prompt": """Based on the context (URL, link text, surrounding elements) for the PDF link below, determine the DOCUMENT_TYPE (e.g., 'User Manual', 'Quick Start Guide', 'Datasheet', 'Specifications', 'Service Manual') and LANGUAGE (e.g., 'EN', 'DE', 'FR', 'Multi', if obvious, otherwise default to 'EN').

Context:
---
{context}
---

Respond ONLY with a valid JSON object: {{"document_type": "...", "language": "..."}}. Use 'Manual' and 'EN' as defaults if unsure.""",
        "filename_template": "{topic}_{document_type}_{language}",
        "required_metadata": ["document_type"],  # Language is helpful but not strictly required
    },
}

# --- Instruction Pack 4: Finding Specific Legal Documents (Example - Requires careful prompting) ---
# NOTE: Legal document searches can be complex due to jurisdiction, specific courts, etc.
# This is a simplified example.
LEGAL_DOCUMENT_INSTRUCTIONS = {
    "search_phase": {
        "search_query_template": '"{topic}" court filing OR legal document OR case text PDF',
        "target_site_identification_prompt": """Analyze the search results summary for "{search_term}".
Identify a URL likely pointing to an official court website (e.g., *.uscourts.gov), legal repository (like CourtListener, RECAP), or official government archive hosting the specific legal case document or docket. Avoid news summaries or law firm analyses unless they directly link to the official document PDF.

Search Results Summary:
---
{search_results_summary}
---

Respond ONLY with JSON: {{"target_url": "URL_or_null"}}.""",
        "search_engine": "google",
    },
    "exploration_phase": {
        "exploration_goal_prompt": "Explore the legal resource website for '{topic}'. Identify and download the relevant court filing, judgment, or legal document PDF.",
        "navigation_keywords": [
            "document",
            "filing",
            "opinion",
            "judgment",
            "docket",
            "case",
            "pdf",
            "download",
            "view",
            "attachment",
            "exhibit",
        ],
        "pdf_keywords": [
            "document",
            "filing",
            "opinion",
            "judgment",
            "pdf",
            "download",
            "attachment",
            "exhibit",
            "order",
        ],
        "pdf_url_patterns": [r"\.pdf(\?|$)", r"/downloadDoc", r"/viewDoc"],
        "max_steps": 15,
    },
    "download_phase": {
        "metadata_extraction_prompt": """Based on the context for the legal document PDF link below, extract the approximate FILING_DATE (YYYY-MM-DD or YYYY) and a short DOCUMENT_CATEGORY (e.g., 'Complaint', 'Motion', 'Opinion', 'Judgment', 'Order', 'Exhibit').

Context:
---
{context}
---

Respond ONLY with JSON: {{"date": "...", "document_category": "..."}}. Use current date or 'Filing' if unknown.""",
        "filename_template": "{date}_{topic}_{document_category}",
        "required_metadata": ["date", "document_category"],
    },
}

# ____________________________________________________________________________________________________________________________________________________________________________________________

# --- Instruction Pack 5: Simple Search Summary ---
SIMPLE_SEARCH_SUMMARY_INSTRUCTIONS = {
    "search_params": {
        "engines": ["google", "duckduckgo"],  # Which engines to use
        "num_results_per_engine": 3,  # How many results to fetch from each
    },
    # Prompt for the LLM that will summarize each page's content
    "summarization_prompt": """Concisely summarize the key information from the following web page content, focusing on its relevance to the search query '{query}'. Output a brief 2-3 sentence summary only.

Page Content:
---
{page_content}
---

Concise Summary:""",
    # Optional: Add filters if needed
    # "url_filter_keywords": [], # e.g., ["blog", "news"] to only summarize blog/news
    # "min_content_length_for_summary": 150 # e.g., skip very short pages
}

# --- Instruction Pack 6: Technical Search Summary ---
TECHNICAL_SEARCH_SUMMARY_INSTRUCTIONS = {
    "search_params": {
        "engines": ["google", "bing"],  # Maybe Bing is better for some technical queries
        "num_results_per_engine": 5,  # Get more results for technical topics
    },
    "summarization_prompt": """Analyze the following web page content related to the technical search query '{query}'. Extract and summarize the core technical concepts, definitions, or conclusions presented. Focus on accuracy and specific details if available. Keep the summary to 3-4 sentences.

Page Content:
---
{page_content}
---

Technical Summary:""",
    "url_filter_keywords": [
        "docs",
        "tutorial",
        "research",
        "arxiv",
        "github",
        "developer",
    ],  # Prioritize technical sources
    "min_content_length_for_summary": 300,  # Expect longer content
}

# ____________________________________________________________________________________________________________________________________________________________________________________________


# --- Instruction Pack 7: Extract Job Posting Details ---
JOB_POSTING_EXTRACTION_INSTRUCTIONS = {
    "data_source": {
        "source_type": "dynamic_crawl",  # Find URLs by crawling
        "crawl_config": {
            "start_url": "https://www.google.com/search?q=software+engineer+jobs+remote",  # Example search
            "list_item_selector": "a[href*='/jobs/']",  # Adjust selector based on actual job board/search results
            "next_page_selector": "#pnnext",  # Google's next page link ID (may change)
            "max_pages_to_crawl": 3,  # Limit crawl depth
            "max_urls_limit": 20,  # Limit total jobs to process
        },
        # Alternatively, provide a list directly:
        # "source_type": "list",
        # "urls": ["https://example-job-board.com/job/123", "https://example-job-board.com/job/456"]
    },
    "extraction_details": {
        # Prompt asking LLM to extract specific fields
        "schema_or_prompt": """From the provided job posting web page content, extract the following details:
- job_title: The official title of the position.
- company_name: The name of the hiring company.
- location: The primary location(s) listed (e.g., "Remote", "New York, NY").
- salary_range: Any mentioned salary range or compensation details (e.g., "$120k - $150k", "Competitive").
- key_skills: A list of the top 3-5 required technical skills or qualifications mentioned.

Web Page Content Context:
---
{page_content}
---

Respond ONLY with a valid JSON object containing these keys. If a field is not found, use null or an empty list for key_skills.""",
        "extraction_llm_model": "openai/gpt-4.1-mini",  # Specify model for extraction
    },
    "output_config": {
        "format": "json_list",  # Output as a list of JSON objects
        "error_handling": "include_error",  # Include URLs that failed in the errors dict
    },
}

# --- Instruction Pack 8: Extract Product Details (Schema Example) ---
ECOMMERCE_PRODUCT_EXTRACTION_INSTRUCTIONS = {
    "data_source": {
        "source_type": "list",
        # URLs would be provided by the calling code/agent based on what products to check
        "urls": [
            # Example URLs (replace with actual ones for testing)
            # "https://www.amazon.com/dp/B08H75RTZ8/", # Example Kindle Paperwhite
            # "https://www.bestbuy.com/site/sony-wh1000xm5-wireless-noise-cancelling-over-the-ear-headphones-black/6505725.p?skuId=6505725"
        ],
    },
    "extraction_details": {
        # Using a JSON schema to define desired output
        "schema_or_prompt": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "The main name or title of the product.",
                },
                "price": {
                    "type": "string",
                    "description": "The current listed price, including currency symbol (e.g., '$149.99').",
                },
                "rating": {
                    "type": "number",
                    "description": "The average customer rating (e.g., 4.7). Null if not found.",
                },
                "num_reviews": {
                    "type": "integer",
                    "description": "The total number of customer reviews. Null if not found.",
                },
                "availability": {
                    "type": "string",
                    "description": "Stock status (e.g., 'In Stock', 'Out of Stock', 'Available for Pre-order').",
                },
            },
            "required": ["product_name", "price", "availability"],
        },
        "extraction_llm_model": "openai/gpt-4.1-mini",  # Use a capable model
    },
    "output_config": {
        "format": "csv_string",  # Output as CSV text
        "error_handling": "skip",  # Skip pages that fail
    },
}


# ____________________________________________________________________________________________________________________________________________________________________________________________


# --- Instruction Pack 9: Login and Check Order Status ---
ORDER_STATUS_WORKFLOW_INSTRUCTIONS = {
    "start_url": "https://the-internet.herokuapp.com/login",  # Example login page
    "workflow_goal_prompt": "Log in using the provided 'username' and 'password', navigate to the secure area, and read the text content of the success message banner.",
    "available_actions": ["type", "click", "read_value", "finish_success", "finish_failure"],
    "llm_model": "openai/gpt-4.1-mini",  # Model for guidance
    "max_steps": 8,
    "input_data_mapping": {  # Maps abstract names to keys in input_data passed to the tool
        "user": "username",
        "pass": "password",
    },
    "element_finding_hints": [
        "username field",
        "password field",
        "login button",
        "success message",
        "logout link",
    ],
    # success_condition_prompt could be added for more complex checks
    # step_prompts are likely not needed for this simple login example
}

# --- Instruction Pack 10: Submit a Simple Contact Form ---
CONTACT_FORM_WORKFLOW_INSTRUCTIONS = {
    "start_url": "https://www.selenium.dev/selenium/web/web-form.html",  # Example form page
    "workflow_goal_prompt": "Fill out the web form using the provided 'name', 'email', and 'message'. Then click the submit button and confirm submission by checking if the page title changes to 'Web form processed'.",
    "available_actions": ["type", "click", "finish_success", "finish_failure"],
    "llm_model": "openai/gpt-4.1-mini",
    "max_steps": 10,
    "input_data_mapping": {
        "contact_name": "name",
        "contact_email": "email",  # Assuming input_data has key "email"
        "contact_message": "message",
    },
    "element_finding_hints": [
        "text input field (my-text)",
        "password input (my-password)",
        "textarea (my-textarea)",
        "submit button",
    ],
    # This workflow implicitly checks success via title change, but an explicit prompt could be added:
    # "success_condition_prompt": "Does the current page title indicate the form was processed successfully (e.g., contains 'processed')?"
}

# ____________________________________________________________________________________________________________________________________________________________________________________________


# --- Instruction Pack 11: Monitor Product Price and Availability ---
PRODUCT_MONITORING_INSTRUCTIONS = {
    "monitoring_targets": [
        {
            "url": "https://www.bestbuy.com/site/sony-wh1000xm5-wireless-noise-cancelling-over-the-ear-headphones-black/6505725.p?skuId=6505725",  # Example URL
            "data_points": [
                {
                    "name": "price",
                    "identifier": ".priceView-hero-price span[aria-hidden='true']",  # CSS selector for price element (INSPECT CAREFULLY!)
                    "extraction_method": "selector",
                    "condition": "changed",  # Alert if price changes from previous_values
                },
                {
                    "name": "availability",
                    "identifier": "button[data-button-state='ADD_TO_CART']",  # Selector for Add to Cart button
                    "extraction_method": "selector",  # We just check existence/text
                    # Condition check via LLM
                    "condition": "llm_eval",
                    "llm_condition_prompt": 'Based on the extracted text/presence of the element (\'Current Value\'), is the product currently available for purchase? Respond {"condition_met": true} if available, {"condition_met": false} otherwise.',
                    # If extraction returns text like "Add to Cart", LLM should say true. If "Sold Out" or None, should say false.
                },
                {
                    "name": "product_title",  # Example LLM extraction
                    "identifier": "Extract the main product title from the page content.",
                    "extraction_method": "llm",
                    "condition": "contains",  # Check if title contains expected keyword
                    "condition_value": "WH-1000XM5",
                },
            ],
        },
        # Add more target product URLs here...
        # { "url": "https://...", "data_points": [...] }
    ],
    "llm_config": {
        "model": "openai/gpt-4.1-mini"  # Model for LLM extraction/evaluation
    },
    "concurrency": {
        "max_concurrent_pages": 2  # Limit concurrency for scraping politeness
    },
    "browser_options": {"headless": True},
}

# --- Instruction Pack 12: Monitor Website Content Section ---
WEBSITE_SECTION_MONITORING_INSTRUCTIONS = {
    "monitoring_targets": [
        {
            "url": "https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pKVGlnQVAB?hl=en-US&gl=US&ceid=US%3Aen",  # Example Google News AI section
            "data_points": [
                {
                    "name": "top_headline_text",
                    "identifier": "h3 > a.gPFEn",  # Selector for top headline link text (INSPECT CAREFULLY!)
                    "extraction_method": "selector",
                    "condition": "changed",  # Alert if the top headline changes
                },
                {
                    "name": "second_headline_text",
                    "identifier": "article:nth-of-type(2) h3 > a.gPFEn",  # Selector for second headline
                    "extraction_method": "selector",
                    "condition": "changed",
                },
            ],
        }
    ],
    "llm_config": {
        "model": "openai/gpt-4.1-mini"  # Not strictly needed if only using selectors
    },
    "concurrency": {"max_concurrent_pages": 3},
    "browser_options": {"headless": True},
}


# ____________________________________________________________________________________________________________________________________________________________________________________________


# --- Instruction Pack 13: Market Trend Summary ---
MARKET_TREND_RESEARCH_INSTRUCTIONS = {
    "research_goal_prompt": "Generate a brief summary of the current market trends for {topic}, based on recent news articles and analysis reports.",
    "search_phase": {
        "search_queries": [
            "{topic} market trends 2024",
            "latest news {topic} industry",
            "{topic} market analysis report",
        ],
        "search_engine": "google",
        "num_search_results_per_query": 5,  # Get a few results per query
    },
    "site_selection_phase": {
        # Prompt to select relevant news/analysis sites
        "selection_prompt": """From the search results for '{topic}', select up to {max_urls} URLs that appear to be recent (within the last year if possible) news articles, market analysis reports, or reputable industry blogs discussing market trends. Avoid forum discussions, product pages, or very old content.

Search Results Context:
---
{search_results_context}
---

Respond ONLY with JSON: {{"selected_urls": ["url1", ...]}}""",
        "max_sites_to_visit": 5,  # Limit how many articles are processed
    },
    "extraction_phase": {
        # Prompt to extract key points related to trends
        "extraction_prompt_or_schema": """Extract the main points, key findings, or trend descriptions related to '{topic}' from the provided web page content. Focus on statements about market direction, growth, challenges, or notable events. Output as a JSON object with a key "key_findings" containing a list of strings (each string is a finding/point).

Web Page Content Context:
---
{page_content}
---

Extracted JSON Data:""",
        "extraction_llm_model": "openai/gpt-4.1-mini",  # Model for extraction
    },
    "synthesis_phase": {
        # Prompt to synthesize the findings into a paragraph
        "synthesis_prompt": """Based on the extracted key findings below regarding '{topic}', write a concise paragraph (3-5 sentences) summarizing the major market trends discussed.

Extracted Information Context:
---
{extracted_information_context}
---

Synthesized Market Trend Summary:""",
        "synthesis_llm_model": "openai/gpt-4.1-mini",  # Model for synthesis
        "report_format_description": "A single paragraph summarizing market trends.",
    },
}

# --- Instruction Pack 14: Competitive Analysis Snippets ---
COMPETITIVE_ANALYSIS_INSTRUCTIONS = {
    "research_goal_prompt": "Gather brief summaries of direct competitors mentioned for the product/service '{topic}' from recent reviews or comparison articles.",
    "search_phase": {
        "search_queries": [
            "{topic} vs competitors",
            "{topic} alternatives review",
            "comparison {topic}",
        ],
        "search_engine": "google",
        "num_search_results_per_query": 8,
    },
    "site_selection_phase": {
        "selection_prompt": """From the search results for '{topic}', select up to {max_urls} URLs that seem to be review sites, comparison articles, or tech news discussing competitors or alternatives to {topic}. Prioritize recent results if possible.

Search Results Context:
---
{search_results_context}
---

Respond ONLY with JSON: {{"selected_urls": ["url1", ...]}}""",
        "max_sites_to_visit": 4,
    },
    "extraction_phase": {
        "extraction_prompt_or_schema": """Identify any direct competitors to '{topic}' mentioned in the provided web page content. For each competitor found, extract its NAME and a brief (1-sentence) summary of how it's compared to {topic} or its key differentiator mentioned.

Web Page Content Context:
---
{page_content}
---

Respond ONLY with a valid JSON object with a key "competitors", where the value is a list of objects, each like {"name": "...", "comparison_summary": "..."}. If no competitors are mentioned, return {"competitors": []}.""",
        "extraction_llm_model": "openai/gpt-4.1-mini",
    },
    "synthesis_phase": {
        "synthesis_prompt": """Consolidate the extracted competitor information related to '{topic}' into a markdown list. For each competitor found across the sources, list its name and a bullet point summary of the comparison points mentioned. Group findings by competitor name.

Extracted Information Context:
---
{extracted_information_context}
---

Consolidated Competitor Markdown List:""",
        "synthesis_llm_model": "openai/gpt-4.1-mini",
        "report_format_description": "A markdown list summarizing mentioned competitors and comparison points.",
    },
}
