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
    # Parâmetros para o chunking
    # O tamanho máximo é 512 para o modelo RoBERTa, mas usamos um pouco menos
    # para dar espaço para tokens especiais ([CLS], [SEP]).
    max_chunk_length = 500
    overlap = 50  # Número de tokens de sobreposição para não cortar entidades

    # Tokeniza o texto inteiro uma vez (mais eficiente)
    tokens = ner_pipeline.tokenizer(text, return_offsets_mapping=True)
    token_count = len(tokens['input_ids'])

    all_entities = []

    print(
        f"Texto grande detectado. Processando em {((token_count - overlap) // (max_chunk_length - overlap)) + 1} chunks...")

    for i in range(0, token_count, max_chunk_length - overlap):
        # Garante que não ultrapassemos o limite da lista de tokens
        end_index = min(i + max_chunk_length, token_count)

        # Seleciona os IDs dos tokens para o chunk atual
        chunk_token_ids = tokens['input_ids'][i:end_index]

        # Converte os IDs de volta para texto para alimentar o pipeline
        # skip_special_tokens=True para remover [CLS] e [SEP] que seriam adicionados de novo
        chunk_text = ner_pipeline.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)

        if not chunk_text.strip():
            continue

        # Executa o pipeline no chunk
        chunk_results = ner_pipeline(chunk_text)

        # Ajusta os offsets (início/fim) para serem relativos ao documento original
        # O offset do primeiro token do chunk nos dá a posição de início no texto original
        start_offset_char = tokens['offset_mapping'][i][0]

        for entity in chunk_results:
            all_entities.append({
                'word': entity['word'],
                'entity_group': entity['entity_group'],
                'score': entity['score'],
                'start': entity['start'] + start_offset_char,
                'end': entity['end'] + start_offset_char
            })

    # --- Desduplicação ---
    # Remove entidades duplicadas que podem ter sido capturadas na sobreposição
    unique_entities = []
    seen_entities = set()
    for entity in sorted(all_entities, key=lambda x: x['start']):
        # Cria um identificador único para a entidade baseado em sua posição e rótulo
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