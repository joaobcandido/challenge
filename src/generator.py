import csv
import hashlib
import io
import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import boto3
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BEDROCK_TOKEN = os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("BEDROCK_API_KEY")
BEDROCK_API_URL = os.getenv("AWS_BEDROCK_API_URL") or "https://bedrock-runtime.us-east-1.amazonaws.com"
BEDROCK_MODEL_ID = os.getenv("AWS_BEDROCK_MODEL_ID") or "anthropic.claude-3-5-sonnet-20241022-v2:0"
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
AWS_PROFILE = os.getenv("AWS_PROFILE") or ""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BEDROCK_COMPAT_API_KEY = os.getenv("BEDROCK_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "openai.gpt-oss-120b"
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE") or "1800")
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP") or "300")
RAG_TOP_K = int(os.getenv("RAG_TOP_K") or "5")
RAG_EMBEDDING_PROVIDER = os.getenv("RAG_EMBEDDING_PROVIDER") or "bedrock"
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL") or "amazon.titan-embed-text-v2:0"
RAG_INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR") or str(BASE_DIR / "data" / "rag_index"))
_OPENAI_COMPAT_INDISPONIVEL = False
_LOGS_CHAMADAS = []
_BEDROCK_RUNTIME_CLIENT = None


def limpar_logs_chamadas() -> None:
    global _OPENAI_COMPAT_INDISPONIVEL
    _LOGS_CHAMADAS.clear()
    _OPENAI_COMPAT_INDISPONIVEL = False


def obter_logs_chamadas() -> list:
    return list(_LOGS_CHAMADAS)


def _registrar_log_chamada(provedor: str, status: str, inicio: float, erro: str = "") -> None:
    _LOGS_CHAMADAS.append({
        "Horário": datetime.now().strftime("%H:%M:%S"),
        "Provedor": provedor,
        "Modelo": (
            _obter_modelo_openai_compat()
            if provedor.startswith("openai")
            else BEDROCK_MODEL_ID if provedor.startswith("bedrock") else "não informado"
        ),
        "Status": status,
        "Duração (s)": round(time.perf_counter() - inicio, 2),
        "Detalhe": erro[:300],
    })


def _obter_modelo_openai_compat() -> str:
    return OPENAI_MODEL.strip() if OPENAI_MODEL else "openai.gpt-oss-120b"


def _obter_cliente_bedrock():
    global _BEDROCK_RUNTIME_CLIENT
    if _BEDROCK_RUNTIME_CLIENT is None:
        sessao = boto3.Session(
            profile_name=AWS_PROFILE or None,
            region_name=AWS_REGION,
        )
        _BEDROCK_RUNTIME_CLIENT = sessao.client("bedrock-runtime")
    return _BEDROCK_RUNTIME_CLIENT

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
        "paginas_fonte": [],
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


def _extrair_paginas_pdf(pdf_file) -> list[dict]:
    if pdf_file is None:
        return []

    buffer = pdf_file.getvalue() if hasattr(pdf_file, "getvalue") else pdf_file.read()
    if not buffer:
        return []

    reader = PdfReader(io.BytesIO(buffer))
    paginas = []
    for numero, pagina in enumerate(reader.pages, start=1):
        texto = pagina.extract_text() or ""
        if texto.strip():
            paginas.append({"pagina": numero, "texto": texto.strip()})
    return paginas


def _extrair_texto_pdf(pdf_file) -> str:
    return "\n\n".join(item["texto"] for item in _extrair_paginas_pdf(pdf_file))


def _normalizar_busca(texto: str) -> str:
    sem_acentos = unicodedata.normalize("NFKD", texto)
    return "".join(char for char in sem_acentos if not unicodedata.combining(char)).lower()


def _dividir_em_chunks(paginas: list[dict]) -> list[dict]:
    chunks = []
    passo = max(1, RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP)
    for pagina in paginas:
        texto = re.sub(r"\s+", " ", pagina["texto"]).strip()
        for inicio in range(0, len(texto), passo):
            trecho = texto[inicio : inicio + RAG_CHUNK_SIZE].strip()
            if trecho:
                chunks.append({
                    "texto": trecho,
                    "pagina": pagina["pagina"],
                    "indice": len(chunks),
                })
            if inicio + RAG_CHUNK_SIZE >= len(texto):
                break
    return chunks


def _tokens_busca(texto: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", _normalizar_busca(texto))
        if token not in {"para", "com", "uma", "dos", "das", "que", "por"}
    }


def _pontuar_chunk(texto_consulta: str, texto_chunk: str) -> float:
    consulta = _tokens_busca(texto_consulta)
    trecho = _tokens_busca(texto_chunk)
    if not consulta or not trecho:
        return 0.0
    cobertura = len(consulta & trecho) / len(consulta)
    densidade = len(consulta & trecho) / len(trecho)
    return (cobertura * 0.8) + (densidade * 0.2)


def _gerar_embedding_bedrock(texto: str) -> list[float]:
    resposta = _obter_cliente_bedrock().invoke_model(
        modelId=RAG_EMBEDDING_MODEL,
        body=json.dumps({"inputText": texto}),
        contentType="application/json",
        accept="application/json",
    )
    dados = json.loads(resposta["body"].read())
    embedding = dados.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("Bedrock não retornou um vetor de embedding válido.")
    return [float(valor) for valor in embedding]


def _gerar_embedding(texto: str) -> list[float] | None:
    provedor = RAG_EMBEDDING_PROVIDER.lower()
    if provedor == "lexical":
        return None
    try:
        if provedor in {"auto", "bedrock"}:
            return _gerar_embedding_bedrock(texto)
        raise ValueError(f"Provedor de embeddings não suportado: {provedor}")
    except Exception as erro:
        logging.warning("Embeddings Bedrock indisponíveis; usando recuperação lexical: %s", erro)
        return None


def _similaridade_vetorial(vetor_a: list[float], vetor_b: list[float]) -> float:
    norma_a = sum(valor * valor for valor in vetor_a) ** 0.5
    norma_b = sum(valor * valor for valor in vetor_b) ** 0.5
    if not norma_a or not norma_b:
        return 0.0
    return sum(a * b for a, b in zip(vetor_a, vetor_b)) / (norma_a * norma_b)


def _criar_indice_rag(pdf_file) -> dict:
    paginas = _extrair_paginas_pdf(pdf_file)
    chunks = _dividir_em_chunks(paginas)
    if not chunks or faiss is None or np is None:
        return {"chunks": chunks, "faiss_index": None}

    pdf_bytes = pdf_file.getvalue() if hasattr(pdf_file, "getvalue") else b""
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    pasta_indice = RAG_INDEX_DIR / digest
    arquivo_indice = pasta_indice / "index.faiss"
    arquivo_metadados = pasta_indice / "metadata.json"

    if arquivo_indice.exists() and arquivo_metadados.exists():
        try:
            indice = faiss.read_index(str(arquivo_indice))
            metadados = json.loads(arquivo_metadados.read_text(encoding="utf-8"))
            if (
                metadados.get("embedding_model") == RAG_EMBEDDING_MODEL
                and metadados.get("chunk_size") == RAG_CHUNK_SIZE
                and metadados.get("chunk_overlap") == RAG_CHUNK_OVERLAP
            ):
                logging.info("Índice RAG reutilizado: %s", pasta_indice)
                return {"chunks": metadados["chunks"], "faiss_index": indice}
        except (OSError, ValueError, json.JSONDecodeError) as erro:
            logging.warning("Não foi possível carregar o índice RAG persistido: %s", erro)

    vetores = [_gerar_embedding(item["texto"]) for item in chunks]
    if any(vetor is None for vetor in vetores):
        return {"chunks": chunks, "faiss_index": None}

    matriz = np.asarray(vetores, dtype="float32")
    faiss.normalize_L2(matriz)
    indice = faiss.IndexFlatIP(matriz.shape[1])
    indice.add(matriz)
    try:
        pasta_indice.mkdir(parents=True, exist_ok=True)
        faiss.write_index(indice, str(arquivo_indice))
        arquivo_metadados.write_text(
            json.dumps({
                "embedding_model": RAG_EMBEDDING_MODEL,
                "chunk_size": RAG_CHUNK_SIZE,
                "chunk_overlap": RAG_CHUNK_OVERLAP,
                "chunks": chunks,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logging.info("Índice RAG persistido: %s", pasta_indice)
    except OSError as erro:
        logging.warning("Não foi possível persistir o índice RAG: %s", erro)
    return {"chunks": chunks, "faiss_index": indice}


def _recuperar_contexto(indice: dict, texto_reclamacao: str) -> tuple[str, list[int]]:
    chunks = indice.get("chunks", [])
    indice_faiss = indice.get("faiss_index")
    if not chunks:
        return "Nenhuma política foi encontrada no PDF.", []

    embedding_consulta = _gerar_embedding(texto_reclamacao)
    resultados = []
    if embedding_consulta and indice_faiss is not None and np is not None:
        consulta = np.asarray([embedding_consulta], dtype="float32")
        faiss.normalize_L2(consulta)
        pontuacoes, posicoes = indice_faiss.search(consulta, min(RAG_TOP_K, len(chunks)))
        resultados = [
            (float(pontuacao), chunks[posicao])
            for pontuacao, posicao in zip(pontuacoes[0], posicoes[0])
            if posicao >= 0
        ]
    else:
        resultados = [
            (_pontuar_chunk(texto_reclamacao, item["texto"]), item)
            for item in chunks
        ]

    resultados.sort(key=lambda resultado: resultado[0], reverse=True)
    selecionados = [
        item for pontuacao, item in resultados[:max(1, RAG_TOP_K)] if pontuacao > 0
    ]
    if not selecionados:
        selecionados = [item for _, item in resultados[:max(1, RAG_TOP_K)]]

    contexto = "\n\n".join(
        f"[Página {item['pagina']}] {item['texto']}" for item in selecionados
    )
    paginas = sorted({item["pagina"] for item in selecionados})
    return contexto, paginas


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
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 800,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    }

    resposta = _obter_cliente_bedrock().invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json",
    )
    dados = json.loads(resposta["body"].read())
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


def _aws_credenciais_disponiveis() -> bool:
    try:
        sessao = boto3.Session(
            profile_name=AWS_PROFILE or None,
            region_name=AWS_REGION,
        )
        return sessao.get_credentials() is not None
    except Exception:
        return False


def _detectar_provedor() -> str:
    if BEDROCK_TOKEN or _aws_credenciais_disponiveis():
        return "bedrock"
    if OPENAI_BASE_URL:
        return "openai-compatible"
    if OPENAI_API_KEY and not OPENAI_BASE_URL:
        return "openai-compatible"
    if BEDROCK_TOKEN:
        return "bedrock"
    return "nenhum"


def _classificar_reclamacao(
    texto_reclamacao: str,
    contexto_politicas: str,
    paginas_fonte: list[int] | None = None,
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
            global _OPENAI_COMPAT_INDISPONIVEL

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
                raise
        elif provedor == "bedrock":
            inicio_chamada = time.perf_counter()
            try:
                resposta_texto = _chamar_bedrock(prompt)
                _registrar_log_chamada("bedrock", "Sucesso", inicio_chamada)
            except Exception as erro_bedrock:
                _registrar_log_chamada("bedrock", "Erro", inicio_chamada, str(erro_bedrock))
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
        dados["paginas_fonte"] = paginas_fonte or []
        return dados
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

            indice_rag = _criar_indice_rag(pdf_file)

        resultados = []
        for indice, linha in enumerate(linhas, start=1):
            texto = _montar_texto_reclamacao(linha)
            texto_reclamacao_original = (
                str(linha.get("texto_reclamacao") or linha.get("texto") or texto).strip()
            )
            contexto_politicas, paginas_fonte = _recuperar_contexto(indice_rag, texto)
            classificacao = _classificar_reclamacao(
                texto,
                contexto_politicas,
                paginas_fonte,
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
                "paginas_fonte": classificacao.get("paginas_fonte", []),
            })

        return resultados
    except Exception as exc:
        logging.error("Erro ao classificar CSV: %s", exc)
        return []


if __name__ == "__main__":
    print("Módulo de classificação de reclamações pronto.")
