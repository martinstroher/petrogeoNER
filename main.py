import os
from transformers import pipeline
import torch

# --- Configuração ---
MODEL_NAME = "hmoreira/xlm-roberta-large-petrogeoner"
FILE_PATH = "extracted_texts.txt"

# Determine if a GPU is available and set the device
device = 0 if torch.cuda.is_available() else -1  # 0 for first GPU, -1 for CPU
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
    """
    Executa o pipeline de NER em um texto longo, dividindo-o em chunks
    com sobreposição para garantir a captura de entidades nas bordas.
    """
    max_chunk_length = 500
    overlap = 50

    tokens = ner_pipeline.tokenizer(text, return_offsets_mapping=True, truncation=False)
    token_count = len(tokens['input_ids'])

    # --- NOVO DIAGNÓSTICO ---
    print(f"DIAGNÓSTICO: Total de caracteres no texto: {len(text)}")
    print(f"DIAGNÓSTICO: Total de tokens detectados: {token_count}")
    num_chunks = ((token_count - overlap) // (max_chunk_length - overlap)) + 1
    print(f"DIAGNÓSTICO: Processando em aproximadamente {num_chunks} chunks...")
    # -----------------------

    all_entities = []

    # Adicionando um contador para ver o progresso
    chunk_counter = 0

    for i in range(0, token_count, max_chunk_length - overlap):
        chunk_counter += 1
        end_index = min(i + max_chunk_length, token_count)
        chunk_token_ids = tokens['input_ids'][i:end_index]

        chunk_text = ner_pipeline.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)

        # --- NOVO DIAGNÓSTICO ---
        if chunk_counter % 100 == 0:  # Imprime o progresso a cada 100 chunks
            print(f"  ... processando chunk {chunk_counter}/{num_chunks}")
        # -----------------------

        if not chunk_text.strip():
            continue

        chunk_results = ner_pipeline(chunk_text)

        if not tokens['offset_mapping']:
            continue

        start_char_offset_index = i
        # Certifique-se de não acessar um índice fora dos limites
        while start_char_offset_index < len(tokens['offset_mapping']) and tokens['offset_mapping'][
            start_char_offset_index] is None:
            start_char_offset_index += 1

        if start_char_offset_index < len(tokens['offset_mapping']):
            start_offset_char = tokens['offset_mapping'][start_char_offset_index][0]

            for entity in chunk_results:
                all_entities.append({
                    'word': entity['word'],
                    'entity_group': entity['entity_group'],
                    'score': entity['score'],
                    'start': entity['start'] + start_offset_char,
                    'end': entity['end'] + start_offset_char
                })

    # ... (o resto da função de desduplicação permanece o mesmo) ...
    # ...
    unique_entities = []
    seen_entities = set()
    for entity in sorted(all_entities, key=lambda x: x['start']):
        entity_id = (entity['start'], entity['end'], entity['entity_group'])
        if entity_id not in seen_entities:
            unique_entities.append(entity)
            seen_entities.add(entity_id)

    return unique_entities

# --- Início da Execução ---

texto_petrolifero = load_text_from_file(FILE_PATH)

if texto_petrolifero:
    print(f"--- Texto Carregado para Análise (Tamanho: {len(texto_petrolifero)} caracteres) ---\n")

    try:
        # Criar o pipeline de Token Classification (NER)
        ner_pipeline = pipeline(
            "ner",
            model=MODEL_NAME,
            aggregation_strategy="first",
            device=device  # Use GPU if available
        )

        # Executar o pipeline com a lógica de chunking
        resultados = ner_with_chunks(texto_petrolifero, ner_pipeline)

        # Exibir os resultados
        print(f"\n--- TOTAL DE ENTIDADES ÚNICAS ENCONTRADAS: {len(resultados)} ---")
        for entity in resultados:
            print(
                f"Entidade: {entity['word']}\n"
                f"  Rótulo: {entity['entity_group']}\n"
                f"  Score: {entity['score']:.4f}\n"
                f"  Início: {entity['start']}, Fim: {entity['end']}\n"
                f"--------------------"
            )

    except Exception as e:
        print(f"ERRO durante a execução do pipeline de NER: {e}")
else:
    print("Análise abortada devido à falha no carregamento do arquivo.")