import os
import csv
from transformers import pipeline
import torch
from tqdm import tqdm  # <-- ADICIONADO: Importar a biblioteca tqdm

# --- Configuração ---
MODEL_NAME = "hmoreira/xlm-roberta-large-petrogeoner"
FILE_PATH = "extracted_texts.txt"
CSV_FILENAME = "resultados_ner.csv"

# Determine if a GPU is available and set the device
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")


def load_text_from_file(filepath):
    """Carrega o conteúdo do arquivo de texto para processamento."""
    if not os.path.exists(filepath):
        print(f"ERRO: O arquivo '{filepath}' não foi encontrado na pasta atual.")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"ERRO ao ler o arquivo: {e}")
        return None


def ner_with_chunks(text, ner_pipeline):
    """Executa o pipeline de NER em um texto longo, dividindo-o em chunks."""
    max_chunk_length = 500
    overlap = 50
    tokens = ner_pipeline.tokenizer(text, return_offsets_mapping=True, truncation=False)
    token_count = len(tokens['input_ids'])

    # Criamos uma lista dos pontos de início de cada chunk para passar ao tqdm
    step = max_chunk_length - overlap
    chunk_start_points = range(0, token_count, step)

    print(f"DIAGNÓSTICO: Processando {len(chunk_start_points)} chunks...")
    all_entities = []

    # <-- MODIFICADO: O loop 'for' agora é envolvido pelo tqdm
    for i in tqdm(chunk_start_points, desc="Processando Chunks", unit="chunk"):
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

    # A desduplicação de entidades por posição permanece a mesma
    unique_entities = []
    seen_entities = set()
    for entity in sorted(all_entities, key=lambda x: x['start']):
        entity_id = (entity['start'], entity['end'], entity['entity_group'])
        if entity_id not in seen_entities:
            unique_entities.append(entity)
            seen_entities.add(entity_id)
    return unique_entities


def collapse_and_aggregate_entities(entities):
    """Agrupa entidades por nome, conta-as e calcula o score médio."""
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
    """Salva uma lista de resultados agregados em um arquivo CSV."""
    if not results:
        print("Nenhum resultado para salvar.")
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
        print(f"\nResultados salvos com sucesso no arquivo '{filename}'")
    except Exception as e:
        print(f"\nERRO ao salvar o arquivo CSV: {e}")


# --- Início da Execução ---
texto_petrolifero = load_text_from_file(FILE_PATH)

if texto_petrolifero:
    print(f"--- Texto Carregado para Análise (Tamanho: {len(texto_petrolifero)} caracteres) ---\n")
    try:
        ner_pipeline = pipeline("ner", model=MODEL_NAME, aggregation_strategy="first", device=device)
        raw_results = ner_with_chunks(texto_petrolifero, ner_pipeline)
        summarized_results = collapse_and_aggregate_entities(raw_results)

        # Exibir os resultados no console
        print(f"\n--- RESUMO DE ENTIDADES ÚNICAS (Total: {len(summarized_results)}) ---")
        for entity in summarized_results:
            print(
                f"Entidade: {entity['entity']}\n  Rótulo: {entity['label']}\n  Contagem: {entity['count']}\n  Score Médio: {entity['avg_score']:.4f}\n--------------------")

        # Salvar os resultados no arquivo CSV
        save_results_to_csv(summarized_results, CSV_FILENAME)

    except Exception as e:
        print(f"ERRO durante a execução do pipeline de NER: {e}")
else:
    print("Análise abortada devido à falha no carregamento do arquivo.")