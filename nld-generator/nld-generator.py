import pandas as pd
import google.generativeai as genai
import os
import time

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    print("API Key do Gemini configurada com sucesso a partir das variáveis de ambiente.")
except KeyError:
    print("ERRO: A variável de ambiente GEMINI_API_KEY não foi encontrada.")
    print("Por favor, configure-a ou insira a chave diretamente no script na variável API_KEY.")
    exit()

MODEL_NAME = "gemini-2.5-pro"

INPUT_FILE = "../output/consolidated_ner_results.csv"
OUTPUT_FILE = "../output/consolidated_ner_results_with_nlds.csv"
OUTPUT_FAILURE_FILE = '../output/unknown_terms.csv'

generation_config = genai.GenerationConfig(
    temperature=0.0,
)

def load_terms_and_labels_from_csv(filepath):
    """Carrega os termos e seus rótulos de um arquivo CSV."""
    if not os.path.exists(filepath):
        print(f"ERRO: O arquivo '{filepath}' não foi encontrado.")
        return None
    try:
        df = pd.read_csv(filepath, encoding='utf-8', delimiter=',', header=0, usecols=['Readable_Term', 'Label'])
        print(f"Sucesso! {len(df)} termos e rótulos carregados de '{filepath}'.")
        return df
    except Exception as e:
        print(f"ERRO ao ler o arquivo CSV: {e}")
        return None


system_instruction_correcao = "You are a data processing assistant specializing in correcting and standardizing technical terms from the geology domain."

prompt_template_correcao = """"Your task is to correct and format the following technical term according to strict geological and petroleum domain standards. Follow these rules precisely:

1. If the term is a concatenated phrase, separate the words (e.g., "carbonatemounds" -> "carbonate mounds").
2. If the term contains an obvious typo, correct it.
3. If the term is already correct and well-formatted, return it unchanged.
4. If the term is nonsensical or unrecognizable return the exact string "UNKNOWN_TERM".

Your response must contain ONLY the corrected term or the "UNKNOWN_TERM" flag.

Term to be corrected:
"{termo_bruto}"
"""

system_instruction_definicao = "You are a senior geoscientist and ontology engineer. Your expertise is in oil and gas exploration geology, with a specific focus on the carbonate reservoirs of the Brazilian Pre-Salt."
prompt_template_definicao = """Generate a concise and precise Natural Language Definition (NLD) for the provided term, using the assigned label as context for disambiguation.

Mandatory Instructions:
1. The definition must strictly follow the Aristotelian structure "X is a Y that Z". For example, "An amount of rock is a solid consolidated earth material that is constituted by an aggregate of particles made of mineral matter or material of biological origin"
2. **Contextual Disambiguation:** You should use the `Label` to resolve any ambiguity in the term. For example, if the `Term to be defined` is "Paraná" and the assigned `Label` is "BACIA", you must define the Paraná Basin, not the river or the state.
3. The definition should be technical yet clear, and a maximum of three sentences.
4. Your response must contain only the generated NLD, without any extra text.

Term to be defined: "{termo_corrigido}"
Assigned Label: "{rotulo_ner}"
"""

df_termos = load_terms_and_labels_from_csv(INPUT_FILE)

if df_termos is not None:
    resultados = []
    termos_para_revisao = []

    model_correcao = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction_correcao,
                                           generation_config=generation_config)
    model_definicao = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction_definicao,
                                            generation_config=generation_config)

    total_termos = len(df_termos)
    for index, row in df_termos.iterrows():
        termo_bruto = row['Readable_Term']
        rotulo_ner = row['Label']

        print(f"Processando termo {index + 1}/{total_termos}: '{termo_bruto}'...")

        try:
            response_correcao = model_correcao.generate_content(
                prompt_template_correcao.format(termo_bruto=termo_bruto))
            termo_corrigido = response_correcao.text.strip()

            if termo_corrigido == "UNKNOWN_TERM" or len(termo_corrigido.split()) > 5:
                motivo = 'Não reconhecido pelo LLM' if termo_corrigido == "UNKNOWN_TERM" else 'Resposta inválida do LLM de correção'
                print(f"  -> Termo '{termo_bruto}' inválido. Marcado para revisão manual. Motivo: {motivo}")
                termos_para_revisao.append({'Termo_Original': termo_bruto, 'Label': rotulo_ner, 'Motivo': motivo, 'Resposta_LLM': termo_corrigido})
                time.sleep(1)
                continue

            response_definicao = model_definicao.generate_content(
                prompt_template_definicao.format(termo_corrigido=termo_corrigido, rotulo_ner=rotulo_ner))
            nld_gerada = response_definicao.text.strip()

            if "não tenho informações" in nld_gerada.lower() or "termo desconhecido" in nld_gerada.lower() or len(
                    nld_gerada) == 0:
                print(f"  -> Definição para '{termo_corrigido}' não encontrada. Marcado para revisão manual.")
                termos_para_revisao.append({'Termo_Original': termo_bruto, 'Termo_Corrigido': termo_corrigido, 'Label': rotulo_ner, 'Motivo': 'Sem definição encontrada'})
            else:
                print(f"  -> Definição gerada com sucesso.")
                resultados.append(
                    {'Termo_Corrigido': termo_corrigido, 'NLD': nld_gerada, 'Rótulo_Original': rotulo_ner})

            time.sleep(1)

        except Exception as e:
            print(f"  -> ERRO ao processar o termo '{termo_bruto}': {e}")
            termos_para_revisao.append({'Termo_Original': termo_bruto, 'Label': rotulo_ner, 'Erro': str(e)})

    print("\nProcessamento concluído. Salvando resultados...")

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"{len(df_resultados)} definições salvas em 'nlds_generated.csv'")

    if termos_para_revisao:
        df_revisao = pd.DataFrame(termos_para_revisao)
        df_revisao.to_csv(OUTPUT_FAILURE_FILE, index=False, encoding='utf-8-sig')
        print(f"{len(df_revisao)} termos para revisão manual salvos em 'termos_para_revisao_manual.csv'")