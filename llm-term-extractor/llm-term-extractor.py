import pandas as pd
import google.generativeai as genai
import os
import json
import time

### 1. CONFIGURATION ###

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    print("Gemini API Key configured successfully.")
except Exception as e:
    print(f"ERROR configuring Gemini API: {e}")
    exit()

# --- User Configuration ---
# This script now splits by the delimiter, so CHUNK_SIZE is no longer needed.
PAPER_DELIMITER = "[END_OF_PAPER]"
INPUT_TXT_FILE = "../resources/extracted_texts_delimited_per_paper.txt"
OUTPUT_CSV_FILE = "../output/llm_extracted_terms.csv"
MODEL_NAME = os.environ["LLM_MODEL_NAME"] # Corrected to a valid model name

generation_config = genai.GenerationConfig(
    temperature=0.0,
    response_mime_type="application/json"
)


### 2. HELPER FUNCTION ###
def load_text_from_file(filepath):
    """Loads the entire content of a text file into a single string."""
    if not os.path.exists(filepath):
        print(f"ERROR: The file '{filepath}' was not found.")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Success! Loaded {len(content)} characters from '{filepath}'.")
        return content
    except Exception as e:
        print(f"ERROR reading the text file: {e}")
        return None


### 3. PROMPT DEFINITION (No changes needed here) ###

system_instruction = "You are an expert geologist and ontology engineer specializing in the South Atlantic Pre-Salt petroleum systems. Your task is to extract key conceptual knowledge from technical documents."

prompt_template = """Your task is to analyze the provided text snippet and extract ALL relevant geological concepts useful for building an ontology.

**METHODOLOGY (Follow Strictly):**
1.  **Extract All Concepts:** Identify and extract all relevant geological concepts. Do not rank or limit the number.
2.  **Normalize Terms:** Return all concepts in English and in their singular form.
3.  **Filter Irrelevant Terms:** You MUST exclude non-geological terms, specific named locations (wells, fields), author names, and company names. Focus on conceptual entities.

**OUTPUT FORMAT:**
Your response MUST BE a valid JSON array of strings.

**Example of output array:**
["Microbial Carbonate", "Diagenesis", "Source Rock", "Structural Trap"]

---
**TEXT SNIPPET TO ANALYZE:**
{chunk_text}
"""

### 4. MAIN EXECUTION (Now processes paper by paper) ###

print("Loading the text corpus...")
full_text = load_text_from_file(INPUT_TXT_FILE)

if full_text:
    all_extracted_terms = []

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_instruction,
        generation_config=generation_config
    )

    # --- KEY CHANGE: Split the text into papers using the delimiter ---
    papers = full_text.split(PAPER_DELIMITER)
    # Filter out any empty strings that might result from splitting (e.g., at the start/end of the file)
    papers = [paper.strip() for paper in papers if paper.strip()]
    num_papers = len(papers)
    print(f"\nText has been split into {num_papers} separate papers.")

    # --- KEY CHANGE: Loop through each paper instead of fixed-size chunks ---
    for i, paper_text in enumerate(papers):
        paper_num = i + 1
        print(f"Processing paper {paper_num}/{num_papers}...")

        # Each paper is now a "chunk"
        try:
            final_prompt = prompt_template.format(chunk_text=paper_text)

            response = model.generate_content(final_prompt)

            # Check for blocked response
            if not response.parts:
                print(
                    f"  -> ERROR: API call for paper {paper_num} was blocked. Reason: {response.prompt_feedback.block_reason}")
                continue

            # Parse the JSON response which is a simple list of terms
            terms_from_paper = json.loads(response.text)
            all_extracted_terms.extend(terms_from_paper)
            print(f"  -> Extracted {len(terms_from_paper)} terms from this paper.")

        except Exception as e:
            print(f"  -> An error occurred processing paper {paper_num}: {e}")

        # Pause between API calls to respect rate limits
        time.sleep(2)

    ### 5. SAVE RAW RESULTS ###
    print("\nExtraction complete. Saving all extracted terms...")

    df_raw_results = pd.DataFrame(all_extracted_terms, columns=['Entidade'])
    df_raw_results.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')

    print(
        f"\nSuccess! A total of {len(all_extracted_terms)} raw terms were extracted and saved to '{OUTPUT_CSV_FILE}'.")
    print(f"\nNEXT STEP: Use this file as input for your consolidation and frequency analysis script.")