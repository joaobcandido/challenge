import csv
import io
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
from openai import OpenAI
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

BEDROCK_TOKEN = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
BEDROCK_API_URL = os.getenv("AWS_BEDROCK_API_URL") or "https://bedrock-runtime.us-east-1.amazonaws.com"
BEDROCK_MODEL_ID = os.getenv("AWS_BEDROCK_MODEL_ID") or "anthropic.claude-3-5-sonnet-20241022-v2:0"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BEDROCK_COMPAT_API_KEY = os.getenv("BEDROCK_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "openai.gpt-oss-120b"
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")
GOOGLE_REQUEST_DELAY_SECONDS = float(os.getenv("GOOGLE_REQUEST_DELAY_SECONDS") or "12")
_OPENAI_COMPAT_INDISPONIVEL = False
_GEMINI_INDISPONIVEL = False
_LOGS_CHAMADAS = []
_ULTIMA_CHAMADA_GEMINI = 0.0


def limpar_logs_chamadas() -> None:
    global _OPENAI_COMPAT_INDISPONIVEL, _GEMINI_INDISPONIVEL, _ULTIMA_CHAMADA_GEMINI
    _LOGS_CHAMADAS.clear()
    _OPENAI_COMPAT_INDISPONIVEL = False
    _GEMINI_INDISPONIVEL = False
    _ULTIMA_CHAMADA_GEMINI = 0.0


def obter_logs_chamadas() -> list:
    return list(_LOGS_CHAMADAS)


def _registrar_log_chamada(provedor: str, status: str, inicio: float, erro: str = "") -> None:
    _LOGS_CHAMADAS.append({
        "Horário": datetime.now().strftime("%H:%M:%S"),
        "Provedor": provedor,
        "Modelo": _obter_modelo_openai_compat() if provedor.startswith("openai") else "gemini-flash-latest",
        "Status": status,
        "Duração (s)": round(time.perf_counter() - inicio, 2),
        "Detalhe": erro[:300],
    })


def _obter_modelo_openai_compat() -> str:
    return OPENAI_MODEL.strip() if OPENAI_MODEL else "openai.gpt-oss-120b"

CATEGORIAS = [
    "Cobrança Indevida",
    "Atendimento",
    "Fraude/Segurança",
    "Produto/Serviço",
    "Cancelamento",
    "Outros",
]

SYSTEM_PROMPT = """
Você é o FinGuard, um analista especialista em reclamações de instituições financeiras.
Sua tarefa é fazer a triagem, avaliação de risco e encaminhamento de cada reclamação com base no texto recebido e nas políticas do PDF.

Instruções:
1. Use apenas o texto do PDF de políticas como referência.
2. Considere linguagem informal, erros de digitação e o canal de origem sem penalizar o cliente.
3. Responda em JSON válido com exatamente estas cinco chaves: categoria, produto, sentimento, urgencia e resumo.
4. categoria deve ser exatamente uma destas opções: Cobrança Indevida, Atendimento, Fraude/Segurança, Produto/Serviço, Cancelamento ou Outros.
5. produto deve ser exatamente uma destas opções: Cartão de Crédito, Conta Corrente, Empréstimo, Investimentos, Seguros ou Não Identificado.
6. sentimento deve ser exatamente uma destas opções: Positivo, Neutro, Negativo ou Crítico.
7. urgencia deve ser exatamente uma destas opções: Baixa, Média, Alta ou Crítica.
8. resumo deve ser padronizado, objetivo e ter 2 ou 3 linhas curtas descrevendo o problema.
9. Não adicione texto extra fora do JSON.
"""


def _normalizar_categoria(categoria: str) -> str:
    mapa = {
        "Cobrança Indevida": "Cobrança Indevida",
        "Unauthorized Charge": "Cobrança Indevida",
        "Atendimento": "Atendimento",
        "Customer Support": "Atendimento",
        "Fraude/Segurança": "Fraude/Segurança",
        "Fraude e Segurança": "Fraude/Segurança",
        "Fraud/Security": "Fraude/Segurança",
        "Produto/Serviço": "Produto/Serviço",
        "Produto ou Serviço": "Produto/Serviço",
        "Product or Service": "Produto/Serviço",
        "Cancelamento": "Cancelamento",
        "Cancelamento e Reembolso": "Cancelamento",
        "Cancellation and Refund": "Cancelamento",
        "Outros": "Outros",
        "Other": "Outros",
    }
    return mapa.get(str(categoria).strip(), "Outros")


def _normalizar_produto(produto: str) -> str:
    mapa = {
        "Cartão de Crédito": "Cartão de Crédito",
        "Cartão de crédito": "Cartão de Crédito",
        "Conta Corrente": "Conta Corrente",
        "Conta corrente": "Conta Corrente",
        "Empréstimo": "Empréstimo",
        "Investimentos": "Investimentos",
        "Seguros": "Seguros",
        "Outro": "Não Identificado",
        "Não Identificado": "Não Identificado",
        "Nao Identificado": "Não Identificado",
    }
    return mapa.get(str(produto).strip(), "Não Identificado")


def _normalizar_sentimento(sentimento: str) -> str:
    mapa = {
        "Positivo": "Positivo",
        "Positive": "Positivo",
        "Neutro": "Neutro",
        "Neutral": "Neutro",
        "Negativo": "Negativo",
        "Negative": "Negativo",
        "Crítico": "Crítico",
        "Critico": "Crítico",
        "Critical": "Crítico",
    }
    return mapa.get(str(sentimento).strip(), "Neutro")


def _normalizar_booleano(valor) -> bool:
    if isinstance(valor, bool):
        return valor
    return str(valor).strip().lower() in {"true", "1", "sim", "yes"}


def _normalizar_urgencia(valor: str) -> str:
    texto = str(valor).strip().lower()
    mapa = {
        "baixa": "Baixa",
        "urgência baixa": "Baixa",
        "urgencia baixa": "Baixa",
        "média": "Média",
        "media": "Média",
        "urgência média": "Média",
        "urgencia media": "Média",
        "alta": "Alta",
        "urgência alta": "Alta",
        "urgencia alta": "Alta",
        "crítica": "Crítica",
        "critica": "Crítica",
        "urgência crítica": "Crítica",
        "urgencia critica": "Crítica",
    }
    return mapa.get(texto, "Média")


def _classificacao_padrao(
    status_analise: str,
    mensagem_analise: str,
    acao_sugerida: str,
) -> dict:
    return {
        "produto": "Não Identificado",
        "produto_identificado": "Não Identificado",
        "categoria": "Outros",
        "sentimento": "Neutro",
        "urgencia": "Média",
        "resumo": mensagem_analise,
        "prioridade": "Média",
        "criticidade": "Média",
        "fraude_identificada": False,
        "violacao_regulatoria": False,
        "escalonar_compliance": False,
        "area_responsavel": "Atendimento",
        "status_analise": status_analise,
        "mensagem_analise": mensagem_analise,
        "motivo": mensagem_analise,
        "acao_sugerida": acao_sugerida,
    }


def _status_do_erro(erro: Exception) -> str:
    erro_texto = str(erro).upper()
    if "503" in erro_texto or "UNAVAILABLE" in erro_texto:
        return "Servidor indisponível"
    if "429" in erro_texto or "RESOURCE_EXHAUSTED" in erro_texto:
        return "Limite de requisições atingido"
    if "401" in erro_texto or "UNAUTHORIZED" in erro_texto or "INVALID_API_KEY" in erro_texto:
        return "Falha de autenticação"
    return "Falha de processamento"


def _extrair_texto_pdf(pdf_file) -> str:
    if pdf_file is None:
        return ""

    buffer = pdf_file.getvalue() if hasattr(pdf_file, "getvalue") else pdf_file.read()
    if not buffer:
        return ""

    reader = PdfReader(io.BytesIO(buffer))
    paginas = []
    for pagina in reader.pages:
        texto = pagina.extract_text() or ""
        if texto.strip():
            paginas.append(texto.strip())
    return "\n\n".join(paginas)


def _montar_texto_reclamacao(row: dict) -> str:
    partes = []
    for chave, valor in row.items():
        if valor is None:
            continue
        texto = str(valor).strip()
        if texto:
            partes.append(f"{chave}: {texto}")
    return "\n".join(partes) or "Reclamação sem texto identificado."


def _parse_json_result(texto: str) -> dict:
    texto = texto.strip()
    if texto.startswith("```"):
        texto = texto.strip("` ")
        if "json" in texto.lower().splitlines()[0].lower():
            linhas = texto.splitlines()[1:]
            texto = "\n".join(linhas)
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        try:
            inicio = texto.find("{")
            fim = texto.rfind("}")
            if inicio != -1 and fim != -1 and fim > inicio:
                return json.loads(texto[inicio : fim + 1])
        except json.JSONDecodeError:
            pass

    return {
        "produto": "Não Identificado",
        "categoria": "Outros",
        "sentimento": "Neutro",
        "urgencia": "Média",
        "resumo": "Não foi possível interpretar a resposta da IA.",
        "prioridade": "Média",
        "motivo": "Não foi possível interpretar a resposta da IA.",
        "acao_sugerida": "Revisar manualmente a reclamação e encaminhar ao setor responsável.",
    }


def _chamar_bedrock(prompt: str) -> str:
    if not BEDROCK_TOKEN:
        raise ValueError("AWS_BEARER_TOKEN_BEDROCK não configurado.")

    url = f"{BEDROCK_API_URL.rstrip('/')}/model/{BEDROCK_MODEL_ID}/invoke"
    headers = {
        "Authorization": f"Bearer {BEDROCK_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 800,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    }

    resposta = requests.post(url, json=payload, headers=headers, timeout=60)
    if resposta.status_code != 200:
        raise RuntimeError(f"Bedrock API falhou: {resposta.status_code} - {resposta.text}")

    dados = resposta.json()
    if "content" in dados and isinstance(dados["content"], list):
        return "".join(bloco.get("text", "") for bloco in dados["content"] if isinstance(bloco, dict))

    if "output" in dados and "text" in dados["output"]:
        return str(dados["output"]["text"])

    return json.dumps(dados, ensure_ascii=False)


def _chamar_openai_compat(prompt: str, modelo: str = "") -> str:
    if not OPENAI_BASE_URL and not OPENAI_API_KEY and not BEDROCK_COMPAT_API_KEY and not BEDROCK_TOKEN:
        raise ValueError("Nenhuma configuração OpenAI-compatible encontrada.")

    if OPENAI_BASE_URL and "bedrock" in OPENAI_BASE_URL.lower():
        api_key_final = BEDROCK_COMPAT_API_KEY or BEDROCK_TOKEN or OPENAI_API_KEY or "dummy"
    else:
        api_key_final = OPENAI_API_KEY or BEDROCK_COMPAT_API_KEY or BEDROCK_TOKEN or "dummy"

    modelo_ativo = modelo.strip() if modelo else _obter_modelo_openai_compat()
    client = OpenAI(
        api_key=api_key_final,
        base_url=OPENAI_BASE_URL,
    )

    payload = {
        "model": modelo_ativo,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }

    if not (OPENAI_BASE_URL and "bedrock" in OPENAI_BASE_URL.lower()):
        payload["response_format"] = {"type": "json_object"}

    logging.info("Usando modelo OpenAI-compatible: %s em %s", modelo_ativo, OPENAI_BASE_URL)

    resposta = client.chat.completions.create(
        **payload,
    )

    return resposta.choices[0].message.content or "{}"


def _chamar_gemini(prompt: str, modelo: str = "gemini-flash-latest") -> str:
    if client is None:
        raise ValueError("GOOGLE_API_KEY não configurada.")

    global _ULTIMA_CHAMADA_GEMINI
    decorrido = time.monotonic() - _ULTIMA_CHAMADA_GEMINI
    espera = GOOGLE_REQUEST_DELAY_SECONDS - decorrido
    if espera > 0:
        logging.info("Aguardando %.1fs antes da próxima chamada Gemini", espera)
        time.sleep(espera)

    resposta = client.models.generate_content(
        model=modelo,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
        ),
    )
    _ULTIMA_CHAMADA_GEMINI = time.monotonic()
    return resposta.text or "{}"


def _detectar_provedor() -> str:
    if OPENAI_BASE_URL:
        return "openai-compatible"
    if OPENAI_API_KEY and not OPENAI_BASE_URL:
        return "openai-compatible"
    if BEDROCK_TOKEN:
        return "bedrock"
    if client is not None:
        return "gemini"
    return "nenhum"


def _classificar_reclamacao(
    texto_reclamacao: str,
    contexto_politicas: str,
    definicao_classificador: str = "",
    provedor_preferido: str = "auto",
    modelo_preferido: str = "",
) -> dict:
    prompt = f"""
    DEFINIÇÃO DO CLASSIFICADOR:
    {definicao_classificador.strip() or "Analise a reclamação com base nas políticas fornecidas."}

    CONTEXTO DAS POLÍTICAS:
    {contexto_politicas[:12000]}

    RECLAMAÇÃO:
    {texto_reclamacao}
    """

    provedor = provedor_preferido if provedor_preferido != "auto" else _detectar_provedor()

    try:
        if provedor == "openai-compatible":
            global _OPENAI_COMPAT_INDISPONIVEL, _GEMINI_INDISPONIVEL
            if _OPENAI_COMPAT_INDISPONIVEL and _GEMINI_INDISPONIVEL:
                raise RuntimeError(
                    "Endpoint Bedrock e fallback Gemini indisponíveis nesta execução."
                )

            try:
                if _OPENAI_COMPAT_INDISPONIVEL:
                    raise RuntimeError("Endpoint OpenAI-compatible indisponível nesta execução.")
                inicio_chamada = time.perf_counter()
                resposta_texto = _chamar_openai_compat(prompt, modelo_preferido)
                _registrar_log_chamada("openai-compatible", "Sucesso", inicio_chamada)
            except Exception as erro_openai:
                _OPENAI_COMPAT_INDISPONIVEL = True
                _registrar_log_chamada(
                    "openai-compatible",
                    "Erro",
                    locals().get("inicio_chamada", time.perf_counter()),
                    str(erro_openai),
                )
                if client is None or _GEMINI_INDISPONIVEL:
                    raise
                logging.warning(
                    "OpenAI-compatible falhou; tentando fallback Gemini uma vez: %s",
                    erro_openai,
                )
                try:
                    provedor = "gemini-fallback"
                    inicio_chamada = time.perf_counter()
                    resposta_texto = _chamar_gemini(prompt, modelo_preferido or "gemini-flash-latest")
                    _registrar_log_chamada("gemini-fallback", "Sucesso", inicio_chamada)
                except Exception:
                    _GEMINI_INDISPONIVEL = True
                    _registrar_log_chamada(
                        "gemini-fallback",
                        "Erro",
                        locals().get("inicio_chamada", time.perf_counter()),
                        "Falha ou limite de quota do Gemini",
                    )
                    raise
        elif provedor == "bedrock":
            inicio_chamada = time.perf_counter()
            try:
                resposta_texto = _chamar_bedrock(prompt)
                _registrar_log_chamada("bedrock", "Sucesso", inicio_chamada)
            except Exception as erro_bedrock:
                _registrar_log_chamada("bedrock", "Erro", inicio_chamada, str(erro_bedrock))
                raise
        elif provedor == "gemini":
            if _GEMINI_INDISPONIVEL:
                raise RuntimeError("Fallback Gemini indisponível nesta execução.")
            inicio_chamada = time.perf_counter()
            try:
                resposta_texto = _chamar_gemini(prompt, modelo_preferido or "gemini-flash-latest")
                _registrar_log_chamada("gemini", "Sucesso", inicio_chamada)
            except Exception as erro_gemini:
                _GEMINI_INDISPONIVEL = True
                _registrar_log_chamada("gemini", "Erro", inicio_chamada, str(erro_gemini))
                raise
        else:
            return _classificacao_padrao(
                "Análise não realizada",
                "Nenhum provedor de IA está configurado.",
                "Configure um provedor de IA ou encaminhe para análise manual.",
            )

        logging.info("Usando provedor de IA: %s", provedor)
        dados = _parse_json_result(resposta_texto)
        dados["produto"] = _normalizar_produto(
            dados.get("produto", dados.get("produto_identificado", "Não Identificado"))
        )
        dados["produto_identificado"] = dados["produto"]
        dados["categoria"] = _normalizar_categoria(dados.get("categoria", "Outros"))
        dados["sentimento"] = _normalizar_sentimento(dados.get("sentimento", "Neutro"))
        dados["urgencia"] = _normalizar_urgencia(
            dados.get("urgencia", dados.get("prioridade", "Média"))
        )
        dados["prioridade"] = dados["urgencia"]
        dados["resumo"] = str(
            dados.get("resumo", "Resumo não informado.")
        ).strip() or "Resumo não informado."
        dados["criticidade"] = dados.get("criticidade", "Média").strip() or "Média"
        dados["fraude_identificada"] = _normalizar_booleano(dados.get("fraude_identificada", False))
        dados["violacao_regulatoria"] = _normalizar_booleano(dados.get("violacao_regulatoria", False))
        dados["escalonar_compliance"] = _normalizar_booleano(
            dados.get("escalonar_compliance", False)
        )
        dados["area_responsavel"] = dados.get("area_responsavel", "Atendimento")
        dados["status_analise"] = "Concluída"
        dados["mensagem_analise"] = "Análise realizada com sucesso."
        dados["motivo"] = dados.get("motivo", "Classificação baseada no contexto das políticas.")
        dados["acao_sugerida"] = dados.get("acao_sugerida", "Analisar o caso com suporte operacional.")
        return dados
    except (ClientError, ServerError, APIError) as e:
        logging.error("Erro ao classificar reclamação com provedor %s: %s", provedor, e)
        status = _status_do_erro(e)
        return _classificacao_padrao(
            status,
            f"{status} ao consultar o provedor {provedor}.",
            "Encaminhar a reclamação para revisão manual do time de atendimento.",
        )
    except Exception as e:
        logging.error("Erro inesperado na classificação com provedor %s: %s", provedor, e)
        status = _status_do_erro(e)
        return _classificacao_padrao(
            status,
            f"{status} durante a análise.",
            "Validar manualmente ou tentar novamente quando o serviço estiver disponível.",
        )


def classificar_reclamacoes_csv(
    csv_file,
    pdf_file,
    definicao_classificador: str = "",
    provedor_preferido: str = "auto",
    modelo_preferido: str = "",
    limite_reclamacoes: int | None = None,
) -> list:
    if csv_file is None:
        return []

    try:
        conteudo_csv = csv_file.read().decode("utf-8-sig", errors="replace")
        leitor = csv.DictReader(io.StringIO(conteudo_csv))

        if leitor.fieldnames is None:
            linhas = [
                {"texto": linha.strip()}
                for linha in conteudo_csv.splitlines()
                if linha.strip()
            ]
        else:
            linhas = list(leitor)

        if limite_reclamacoes is not None:
            linhas = linhas[:max(0, limite_reclamacoes)]

        contexto_politicas = _extrair_texto_pdf(pdf_file)

        resultados = []
        for indice, linha in enumerate(linhas, start=1):
            texto = _montar_texto_reclamacao(linha)
            texto_reclamacao_original = (
                str(linha.get("texto_reclamacao") or linha.get("texto") or texto).strip()
            )
            classificacao = _classificar_reclamacao(
                texto,
                contexto_politicas,
                definicao_classificador,
                provedor_preferido,
                modelo_preferido,
            )
            resultados.append({
                "linha": indice,
                "id": linha.get("id", ""),
                "data_reclamacao": linha.get("data_reclamacao", ""),
                "canal": linha.get("canal", ""),
                "produto": linha.get("produto", ""),
                "status": linha.get("status", ""),
                "texto_reclamacao": texto_reclamacao_original,
                "produto": classificacao.get("produto", "Não Identificado"),
                "produto_identificado": classificacao.get("produto_identificado", "Não Identificado"),
                "categoria": classificacao.get("categoria", "Outros"),
                "sentimento": classificacao.get("sentimento", "Neutro"),
                "urgencia": classificacao.get("urgencia", "Média"),
                "resumo": classificacao.get("resumo", "Resumo não informado."),
                "prioridade": classificacao.get("prioridade", "Média"),
                "criticidade": classificacao.get("criticidade", "Média"),
                "fraude_identificada": classificacao.get("fraude_identificada", False),
                "violacao_regulatoria": classificacao.get("violacao_regulatoria", False),
                "escalonar_compliance": classificacao.get("escalonar_compliance", False),
                "area_responsavel": classificacao.get("area_responsavel", "Atendimento"),
                "status_analise": classificacao.get("status_analise", "Análise não realizada"),
                "mensagem_analise": classificacao.get("mensagem_analise", "Análise não realizada."),
                "motivo": classificacao.get("motivo", "Classificação automática."),
                "acao_sugerida": classificacao.get("acao_sugerida", "Revisar manualmente."),
            })

        return resultados
    except Exception as exc:
        logging.error("Erro ao classificar CSV: %s", exc)
        return []


if __name__ == "__main__":
    print("Módulo de classificação de reclamações pronto.")
