from transformers import pipeline

# 1. Defina o nome do modelo
model_name = "hmoreira/xlm-roberta-large-petrogeoner"

# 2. Crie o pipeline de Token Classification (NER)
# Isso carrega automaticamente o tokenizer e o modelo corretos
ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="first")

# 3. Seu texto petrolífero
texto_petrolifero = (
    "A Bacia de Campos possui o Campo de Roncador, onde o reservatório é composto "
    "por arenito da Formação Macaé, uma unidade litoestratigráfica do Cretáceo."
)

# 4. Execute o pipeline no seu texto
resultados = ner_pipeline(texto_petrolifero)

print(resultados)

# 5. Exiba os resultados
for entity in resultados:
    print(
        f"Entidade: {entity['word']}\n"
        f"  Rótulo: {entity['entity_group']}\n"
        f"  Score: {entity['score']:.4f}\n"
        f"--------------------"
    )