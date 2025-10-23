import pandas as pd
import google.generativeai as genai
import os
import time
import json  # Usaremos a biblioteca JSON

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    print("Gemini API Key configured successfully.")
except Exception as e:
    print(f"ERROR configuring Gemini API: {e}")
    exit()

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 10))
MODEL_NAME = "gemini-2.5-pro"
INPUT_FILE_PATH = "../output/consolidated_ner_results_with_nlds.csv"
OUTPUT_FILE_PATH = "../output/categorized_ner_terms.csv"
GEORESERVOIR_DEFS_PATH = "../resources/georeservoir-definitions.txt"
GEOCORE_DEFS_PATH = "../resources/geocore-definitions.txt"
BFO_DEFS_PATH = "../resources/bfo-definitions.txt"

print(f"Processing in batches of {BATCH_SIZE} terms.")

generation_config = genai.GenerationConfig(
    temperature=0.0,
    response_mime_type="application/json"
)


def load_definitions_from_file(filepath):
    """Loads text content from a specified file."""
    if not os.path.exists(filepath):
        print(f"ERROR: Definition file not found at '{filepath}'")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"ERROR reading definition file '{filepath}': {e}")
        return None


georeservoir_definitions= load_definitions_from_file(GEORESERVOIR_DEFS_PATH)
geocore_definitions = load_definitions_from_file(GEOCORE_DEFS_PATH)
bfo_definitions = load_definitions_from_file(BFO_DEFS_PATH)
if not geocore_definitions or not bfo_definitions:
    exit()


def load_nlds_from_csv(filepath):
    """Loads terms, NLDs, and labels from a CSV file."""
    if not os.path.exists(filepath):
        print(f"ERROR: The file '{filepath}' was not found.")
        return None
    try:
        df = pd.read_csv(filepath, encoding='utf-8', delimiter=',', header=0,
                         usecols=['Termo_Corrigido', 'NLD', 'Rótulo_Original'])
        print(f"Success! {len(df)} terms, NLDs, and labels loaded from '{filepath}'.")
        return df
    except Exception as e:
        print(f"ERROR reading the CSV file: {e}")
        return None


system_instruction = "You are an expert ontology engineer specializing in foundational (BFO) and geological (GeoCore and GeoReservoir) ontologies. You process data in batches and your response format MUST be a valid JSON array of objects."
prompt_template = """Your task is to classify a batch of geological terms based on their Natural Language Definitions (NLDs).

**METHODOLOGY (Follow Strictly for each item):**
1.  **Analyze Data:** Read the Term and its NLD.
2.  **Prioritize GeoReservoir:** First, attempt to classify the term into one of the `### GeoReservoir Categories`.
3.  **Fallback to GeoCore:** If and only if no GeoReservoir category is a good fit, then attempt to classify it into one of the `### GeoCore Categories`.
4.  **Fallback to BFO:** If and only if no GeoCore category fits, then attempt to classify it into one of the `### BFO Categories`.
5.  **Final Fallback:** If the term does not fit well into ANY of the provided categories (GeoReservoir, GeoCore, or BFO), you MUST use the string `NOT_CLASSIFIED`.
6.  **Provide Reasoning:** In one short sentence, explain WHY you chose that category based on the NLD.

**INPUT/OUTPUT FORMAT:**
-   **INPUT:** A JSON array of objects, where each object has an "term" and "nld" field.
-   **OUTPUT:** Your response MUST BE a valid JSON array. Each object in the array must contain the "term", the assigned "category", and a "reasoning" string.

---
**ONTOLOGY CATEGORIES REFERENCE:**

### GeoReservoir Categories:
{georeservoir_definitions}

### GeoCore Categories:
{geocore_definitions}

### BFO Categories:
{bfo_definitions}

---
**DATA TO CLASSIFY:**
{json_batch}
"""

df_nlds = load_nlds_from_csv(INPUT_FILE_PATH)

if df_nlds is not None:
    classification_results = []
    model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction,
                                  generation_config=generation_config)
    total_terms = len(df_nlds)

    for i in range(0, total_terms, BATCH_SIZE):
        batch_df = df_nlds.iloc[i:i + BATCH_SIZE]

        batch_list = []
        for index, row in batch_df.iterrows():
            batch_list.append({
                "term": row['Termo_Corrigido'],
                "nld": row['NLD']
            })
        json_batch_str = json.dumps(batch_list, indent=2)

        print(f"Classifying batch of terms {i + 1}-{min(i + BATCH_SIZE, total_terms)} of {total_terms}...")
        try:
            final_prompt = prompt_template.format(geocore_definitions=geocore_definitions,
                                                  bfo_definitions=bfo_definitions,
                                                  georeservoir_definitions= georeservoir_definitions,
                                                  json_batch=json_batch_str)
            response = model.generate_content(final_prompt)
            response_json = json.loads(response.text)

            if len(response_json) != len(batch_df):
                raise ValueError("LLM response length does not match batch size.")

            for idx, result_item in enumerate(response_json):
                original_row = batch_df.iloc[idx]

                classification_results.append({
                    'Term': original_row['Termo_Corrigido'],
                    'Category': result_item['category'],
                    'Original_Label': original_row['Rótulo_Original'],
                    'Reasoning': result_item['reasoning'],
                    'NLD': original_row['NLD']
                })
            print("  -> Batch classified and saved successfully.")

        except json.JSONDecodeError:
            print(f"  -> ERROR: LLM returned invalid JSON. Batch flagged for review.")
        except Exception as e:
            print(f"  -> ERROR classifying batch: {e}. Batch flagged for review.")

        time.sleep(2)

    print("\nClassification complete. Saving results...")

    output_dir = os.path.dirname(OUTPUT_FILE_PATH)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"ERROR creating directory {output_dir}: {e}")
            exit()

    try:
        final_df = pd.DataFrame(classification_results)

        column_order = ['Term', 'Category', 'Reasoning', 'Original_Label', 'NLD']
        final_df = final_df[column_order]  # Reorder columns

        final_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
        print(f"Classification results successfully saved to '{OUTPUT_FILE_PATH}'")
    except Exception as e:
        print(f"ERROR saving results to CSV file '{OUTPUT_FILE_PATH}': {e}")