import os
import csv
from transformers import pipeline
import torch
from tqdm import tqdm

MODEL_NAME = "hmoreira/xlm-roberta-large-petrogeoner"
INPUT_FILE = "../resources/extracted_texts.txt"
OUTPUT_FILE = "../output/ner_results.csv"

device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")


def load_text_from_file(filepath):
    if not os.path.exists(filepath):
        print(f"ERROR: File '{filepath}' not found.")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return None


def ner_with_chunks(text, ner_pipeline):
    max_chunk_length = 500
    overlap = 50
    tokens = ner_pipeline.tokenizer(text, return_offsets_mapping=True, truncation=False)
    token_count = len(tokens['input_ids'])

    step = max_chunk_length - overlap
    chunk_start_points = range(0, token_count, step)

    print(f"Processing {len(chunk_start_points)} chunks...")
    all_entities = []

    for i in tqdm(chunk_start_points, desc="Processing Chunks", unit="chunk"):
        end_index = min(i + max_chunk_length, token_count)
        chunk_token_ids = tokens['input_ids'][i:end_index]
        chunk_text = ner_pipeline.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)

        if not chunk_text.strip(): continue

        chunk_results = ner_pipeline(chunk_text)

        if not tokens['offset_mapping']: continue

        start_char_offset_index = i
        while start_char_offset_index < len(tokens['offset_mapping']) and tokens['offset_mapping'][
            start_char_offset_index] is None:
            start_char_offset_index += 1

        if start_char_offset_index < len(tokens['offset_mapping']):
            start_offset_char = tokens['offset_mapping'][start_char_offset_index][0]
            for entity in chunk_results:
                all_entities.append(
                    {'word': entity['word'], 'entity_group': entity['entity_group'], 'score': entity['score'],
                     'start': entity['start'] + start_offset_char, 'end': entity['end'] + start_offset_char})

    unique_entities = []
    seen_entities = set()
    for entity in sorted(all_entities, key=lambda x: x['start']):
        entity_id = (entity['start'], entity['end'], entity['entity_group'])
        if entity_id not in seen_entities:
            unique_entities.append(entity)
            seen_entities.add(entity_id)
    return unique_entities


def collapse_and_aggregate_entities(entities):
    aggregated_results = {}
    for entity in entities:
        entity_text, entity_score, entity_label = entity['word'], entity['score'], entity['entity_group']
        if entity_text not in aggregated_results:
            aggregated_results[entity_text] = {'scores': [entity_score], 'label': entity_label}
        else:
            aggregated_results[entity_text]['scores'].append(entity_score)
    final_list = []
    for entity_text, data in aggregated_results.items():
        count = len(data['scores'])
        avg_score = sum(data['scores']) / count
        final_list.append({'entity': entity_text, 'label': data['label'], 'count': count, 'avg_score': avg_score})
    final_list.sort(key=lambda x: x['count'], reverse=True)
    return final_list


def save_results_to_csv(results, filename):
    if not results:
        print("No result to save.")
        return
    fieldnames = ['Entidade', 'Rótulo', 'Contagem', 'Score Médio']
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow({
                    'Entidade': row['entity'],
                    'Rótulo': row['label'],
                    'Contagem': row['count'],
                    'Score Médio': f"{row['avg_score']:.4f}".replace('.', ',')
                })
        print(f"\nResults succesfully save in file: '{filename}'")
    except Exception as e:
        print(f"\nERROR saving CSV file: {e}")


text = load_text_from_file(INPUT_FILE)

if text:
    print(f"--- Text loaded for analysis (Size: {len(text)} chars) ---\n")
    try:
        ner_pipeline = pipeline("ner", model=MODEL_NAME, aggregation_strategy="first", device=device)
        raw_results = ner_with_chunks(text, ner_pipeline)
        summarized_results = collapse_and_aggregate_entities(raw_results)

        print(f"\n--- NUMBER OF UNIQUE ENTITIES (Total: {len(summarized_results)}) ---")
        for entity in summarized_results:
            print(
                f"Entity: {entity['entity']}\n  Label: {entity['label']}\n  Count: {entity['count']}\n  Average score: {entity['avg_score']:.4f}\n--------------------")

        save_results_to_csv(summarized_results, OUTPUT_FILE)

    except Exception as e:
        print(f"ERROR during NER pipeline execution: {e}")
else:
    print("Aborting analysis due to error while loading file.")