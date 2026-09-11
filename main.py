import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# 1. Carrega as variáveis de ambiente (.env)
load_dotenv()

# Repassa a chave de ambiente caso esteja salva como BEDROCK_API_KEY
if not os.getenv("OPENAI_API_KEY") and os.getenv("BEDROCK_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("BEDROCK_API_KEY")

# 2. Leitura das Regras da Empresa (PDF)
nome_pdf = "politica_empresa.pdf"
regras_politica = ""

try:
    reader = PdfReader(nome_pdf)
    for i, page in enumerate(reader.pages):
        texto = page.extract_text()
        if texto:
            regras_politica += f"\n--- Página {i + 1} ---\n" + texto
except FileNotFoundError:
    print(f"Erro: O arquivo '{nome_pdf}' não foi encontrado.")
    exit(1)

# 3. Leitura da Base de Reclamações (CSV)
nome_csv = "reclamacoes.csv"

try:
    df_reclamacoes = pd.read_csv(nome_csv, encoding="utf-8")
except FileNotFoundError:
    print(f"Erro: O arquivo '{nome_csv}' não foi encontrado.")
    exit(1)

# 4. Inicialização do Cliente OpenAI
client = OpenAI()

# 5. Prompt do Sistema configurado para Classificação
system_prompt = f"""Você é um auditor especialista em atendimento ao cliente. 
Sua tarefa é analisar as reclamações dos clientes e classificá-las de acordo com as Políticas Internas da Empresa.

REGRAS DA EMPRESA (PDF):
{regras_politica}

Para cada reclamação recebida, retorne:
- Regra Violada/Aplicável (Número e Nome da Regra)
- Procedência: (Procedente / Improcedente / Requer Análise)
- Ação Recomendada: Uma breve orientação de como resolver o problema do cliente.
"""

print("--- INICIANDO ANÁLISE DAS RECLAMAÇÕES ---\n")

# 6. Processamento e Classificação de cada reclamação
for index, row in df_reclamacoes.iterrows():
    reclamacao_texto = (
        f"ID: {row['id']}\n"
        f"Cliente: {row['cliente']}\n"
        f"Categoria: {row['categoria']}\n"
        f"Reclamação: {row['texto_reclamacao']}"
    )

    try:
        response = client.chat.completions.create(
            model="openai.gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analise a seguinte reclamação:\n\n{reclamacao_texto}"}
            ],
            temperature=0.1
        )

        print(f"=== Reclamação #{row['id']} - {row['cliente']} ===")
        print(response.choices[0].message.content)
        print("-" * 50 + "\n")

    except Exception as e:
        print(f"Erro ao processar a reclamação #{row['id']}: {e}")
