import os
from transformers import pipeline

# 1. Configuração
MODEL_NAME = "hmoreira/xlm-roberta-large-petrogeoner"
FILE_PATH = "extracted_texts.txt"


# 2. Função para carregar o texto do arquivo
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


# --- Início da Execução ---

# 3. Carregar o texto
texto_petrolifero = load_text_from_file(FILE_PATH)

if texto_petrolifero:
    print(f"--- Texto Carregado para Análise -------------------------------------\n")

    try:
        # 4. Criar o pipeline de Token Classification (NER)
        # Usamos aggregation_strategy="first" para garantir o agrupamento correto de subpalavras
        ner_pipeline = pipeline(
            "ner",
            model=MODEL_NAME,
            aggregation_strategy="first"
        )

        # 5. Executar o pipeline no texto
        resultados = ner_pipeline(texto_petrolifero)

        # 6. Exibir os resultados
        print("--- RESULTADOS DETALHADOS DO NER ---")
        # print(resultados) # Descomente para ver a estrutura completa

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
