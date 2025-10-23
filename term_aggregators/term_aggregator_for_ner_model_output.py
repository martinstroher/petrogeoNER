import pandas as pd
from collections import Counter
import nltk
import os
from nltk.stem import RSLPStemmer

INPUT_FILE = "../output/ner_results.csv"
OUTPUT_FILE = "../output/consolidated_ner_results.csv"

def load_terms_and_labels_from_csv(filepath):
    """Loads terms and their corresponding labels from a CSV file."""
    if not os.path.exists(filepath):
        print(f"ERROR: The file '{filepath}' was not found.")
        return None
    try:
        # <-- MUDANÇA 2: Ler as colunas do arquivo de entrada: 'Entidade' e 'Rótulo'
        df = pd.read_csv(filepath, encoding='utf-8', delimiter=',', header=0, usecols=['Entidade', 'Rótulo'])
        print(f"Success! {len(df)} terms and labels loaded from '{filepath}'.")
        return df
    except Exception as e:
        print(f"ERROR reading the CSV file: {e}")
        return None

pt_stemmer = RSLPStemmer()

terms_df = load_terms_and_labels_from_csv(INPUT_FILE)

if terms_df is not None:
    stemmed_terms = []
    stem_to_info_map = {}

    print("Starting normalization, stemming, and mapping...")
    for index, row in terms_df.iterrows():
        original_term = row['Entidade']
        label = row['Rótulo']

        if not isinstance(original_term, str) or not isinstance(label, str):
            continue

        clean_original_term = original_term.strip().lower()
        clean_label = label.strip()

        words = clean_original_term.split()
        stemmed_words = [pt_stemmer.stem(p) for p in words]
        final_stem = " ".join(stemmed_words)

        stemmed_terms.append(final_stem)

        if final_stem not in stem_to_info_map:
            stem_to_info_map[final_stem] = {
                'original': clean_original_term,
                'labels': {clean_label}
            }
        else:
            stem_to_info_map[final_stem]['labels'].add(clean_label)
            if len(clean_original_term) < len(stem_to_info_map[final_stem]['original']):
                stem_to_info_map[final_stem]['original'] = clean_original_term

    print("Processing complete.")

    stem_frequencies = Counter(stemmed_terms)

    final_results = []
    for stem, count in stem_frequencies.most_common():
        info = stem_to_info_map.get(stem, {'original': stem, 'labels': {'unknown'}})
        readable_term = info['original']
        final_labels_str = " | ".join(sorted(list(info['labels'])))
        final_results.append((readable_term, final_labels_str, count))

    print("\n--- Most Common Terms (with multiple labels) ---")
    for term, label, count in final_results[:15]:
        print(f"Term: '{term}' | Label: {label} | Count: {count}")

    output_filename = OUTPUT_FILE
    final_df = pd.DataFrame(final_results, columns=['Readable_Term', 'Label', 'Frequency'])
    final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\nFinal results successfully saved to '{output_filename}'")