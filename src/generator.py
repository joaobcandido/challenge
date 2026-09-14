import csv
import hashlib
import io
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gemini-1.5-flash-latest"

RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE") or "1800")
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP") or "300")
RAG_TOP_K = int(os.getenv("RAG_TOP_K") or "5")
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL") or "text-embedding-004"
RAG_INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR") or str(BASE_DIR / "data" / "rag_index"))
GOOGLE_REQUEST_DELAY_SECONDS = float(os.getenv("GOOGLE_REQUEST_DELAY_SECONDS") or "12")

_LOGS_CHAMADAS = []
_ULTIMA_CHAMADA_GEMINI = 0.0


def limpar_logs_chamadas() -> None:
    global _ULTIMA_CHAMADA_GEMINI
    _LOGS_CHAMADAS.clear()
    _ULTIMA_CHAMADA_GEMINI = 0.0


def obter_logs_chamadas() -> list:
    return list(_LOGS_CHAMADAS)


def _registrar_log_chamada(provedor: str, status: str, inicio: float, erro: str = "") -> None:
    _LOGS_CHAMADAS.append({
        "Horário": datetime.now().strftime("%H:%M:%S"),
        "Provedor": provedor,
        "Modelo": OPENAI_MODEL,
        "Status": status,
        "Duração (s)": round(time.perf_counter() - inicio, 2),
        "Detalhe": erro[:300],
    })


CATEGORIAS = [
    "Cobrança Indevida", "Atendimento", "Fraude/Segurança",
    "Produto/Serviço", "Cancelamento", "Outros",
]

SYSTEM_PROMPT = """
Você é o FinGuard, um analista especialista em reclamações de instituições financeiras.
Sua tarefa é fazer a triagem e avaliação de risco com base nas políticas do PDF.

Instruções:
1. Responda estritamente em JSON válido com exatamente estas cinco chaves: categoria, produto, sentimento, urgencia e resumo.
2. categoria: Cobrança Indevida, Atendimento, Fraude/Segurança, Produto/Serviço, Cancelamento ou Outros.
3. produto: Cartão de Crédito, Conta Corrente, Empréstimo, Investimentos, Seguros ou Não Identificado.
4. sentimento: Positivo, Neutro, Negativo ou Crítico.
5. urgencia: Baixa, Média, Alta ou Crítica.
6. resumo: Escreva um texto fluido composto por 2 a 3 frases completas descrevendo o problema ocorrido, o impacto gerado para o cliente e a necessidade de solução. NÃO utilize quebras de linha ou o caractere \\n.
7. Não adicione nenhum texto, markdown ou comentários fora do JSON.
"""


def _normalizar_categoria(categoria: str) -> str:
    mapa = {
        "Cobrança Indevida": "Cobrança Indevida", "Unauthorized Charge": "Cobrança Indevida",
        "Atendimento": "Atendimento", "Customer Support": "Atendimento",
        "Fraude/Segurança": "Fraude/Segurança", "Fraude e Segurança": "Fraude/Segurança",
        "Produto/Serviço": "Produto/Serviço", "Cancelamento": "Cancelamento",
        "Outros": "Outros", "Other": "Outros",
    }
    return mapa.get(str(categoria).strip(), "Outros")


def _normalizar_produto(produto: str) -> str:
    mapa = {
        "Cartão de Crédito": "Cartão de Crédito", "Conta Corrente": "Conta Corrente",
        "Empréstimo": "Empréstimo", "Investimentos": "Investimentos", "Seguros": "Seguros",
    }
    return mapa.get(str(produto).strip(), "Não Identificado")


def _normalizar_sentimento(sentimento: str) -> str:
    mapa = {
        "Positivo": "Positivo", "Neutro": "Neutro", "Neutral": "Neutro",
        "Negativo": "Negativo", "Crítico": "Crítico", "Critical": "Crítico",
    }
    return mapa.get(str(sentimento).strip(), "Neutro")


def _normalizar_urgencia(valor: str) -> str:
    texto = str(valor).strip().lower()
    mapa = {
        "baixa": "Baixa", "urgência baixa": "Baixa",
        "média": "Média", "urgência média": "Média",
        "alta": "Alta", "urgência alta": "Alta",
        "crítica": "Crítica", "urgência crítica": "Crítica",
    }
    return mapa.get(texto, "Média")


def _classificacao_padrao(status_analise: str, mensagem_analise: str, acao_sugerida: str, paginas_fonte: list = None) -> dict:
    return {
        "produto_identificado": "Não Identificado", "categoria": "Outros", "sentimento": "Neutro",
        "urgencia": "Média", "resumo": mensagem_analise, "prioridade": "Média", "criticidade": "Média",
        "fraude_identificada": False, "violacao_regulatoria": False, "escalonar_compliance": False,
        "area_responsavel": "Atendimento", "status_analise": status_analise, 
        "mensagem_analise": mensagem_analise, "motivo": mensagem_analise, "acao_sugerida": acao_sugerida,
        "paginas_fonte": paginas_fonte or [],
    }


def _status_do_erro(erro: Exception) -> str:
    erro_texto = str(erro).upper()
    if "503" in erro_texto or "UNAVAILABLE" in erro_texto: return "Servidor indisponível"
    if "429" in erro_texto or "RESOURCE_EXHAUSTED" in erro_texto: return "Limite de requisições atingido"
    if "401" in erro_texto or "UNAUTHORIZED" in erro_texto: return "Falha de autenticação"
    return "Falha de processamento"


def _extrair_paginas_pdf(pdf_file) -> list[dict]:
    if pdf_file is None: return []
    if hasattr(pdf_file, "seek"):
        pdf_file.seek(0)
    buffer = pdf_file.getvalue() if hasattr(pdf_file, "getvalue") else pdf_file.read()
    if not buffer: return []
    reader = PdfReader(io.BytesIO(buffer))
    paginas = []
    for numero, pagina in enumerate(reader.pages, start=1):
        texto = pagina.extract_text() or ""
        paginas.append({"pagina": numero, "texto": texto.strip() or f"Página {numero}"})
    return paginas


def _dividir_em_chunks(paginas: list[dict]) -> list[dict]:
    chunks = []
    passo = max(1, RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP)
    for pagina in paginas:
        texto = re.sub(r"\s+", " ", pagina["texto"]).strip()
        for inicio in range(0, len(texto), passo):
            trecho = texto[inicio : inicio + RAG_CHUNK_SIZE].strip()
            if trecho:
                chunks.append({"texto": trecho, "pagina": pagina["pagina"], "indice": len(chunks)})
            if inicio + RAG_CHUNK_SIZE >= len(texto): break
    return chunks


def _gerar_embedding(texto: str) -> list[float] | None:
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY não configurada. Impossível gerar embeddings.")
        return None
        
    global _ULTIMA_CHAMADA_GEMINI
    decorrido = time.monotonic() - _ULTIMA_CHAMADA_GEMINI
    espera = (GOOGLE_REQUEST_DELAY_SECONDS / 2) - decorrido
    if espera > 0:
        time.sleep(espera)

    try:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        resposta = client.embeddings.create(input=[texto], model=RAG_EMBEDDING_MODEL)
        _ULTIMA_CHAMADA_GEMINI = time.monotonic()
        return resposta.data[0].embedding
    except Exception as erro:
        logging.error("Erro ao gerar embedding com Gemini: %s", erro)
        return None


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
            if metadados.get("embedding_model") == RAG_EMBEDDING_MODEL:
                logging.info("Índice RAG reutilizado do cache: %s", pasta_indice)
                return {"chunks": metadados["chunks"], "faiss_index": indice}
        except Exception as erro:
            logging.warning("Não foi possível carregar índice em cache: %s", erro)

    logging.info("Gerando novos embeddings para o PDF (%d chunks)...", len(chunks))
    vetores = [_gerar_embedding(item["texto"]) for item in chunks]
    if any(vetor is None for vetor in vetores):
        logging.error("Falha ao gerar um ou mais embeddings para o índice RAG.")
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
                "chunks": chunks,
            }, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logging.info("Índice RAG criado e persistido com sucesso.")
    except OSError as erro:
        logging.warning("Não foi possível persistir o índice RAG: %s", erro)
        
    return {"chunks": chunks, "faiss_index": indice}


def _recuperar_contexto(indice: dict, texto_reclamacao: str) -> tuple[str, list[int]]:
    chunks = indice.get("chunks", [])
    indice_faiss = indice.get("faiss_index")
    
    if not chunks or indice_faiss is None or np is None:
        return "Políticas não carregadas ou FAISS indisponível.", []

    embedding_consulta = _gerar_embedding(texto_reclamacao)
    if not embedding_consulta:
        return "Falha ao gerar embedding da reclamação para busca de contexto.", []

    consulta = np.asarray([embedding_consulta], dtype="float32")
    faiss.normalize_L2(consulta)
    pontuacoes, posicoes = indice_faiss.search(consulta, min(RAG_TOP_K, len(chunks)))
    
    resultados = [
        (float(pontuacao), chunks[posicao])
        for pontuacao, posicao in zip(pontuacoes[0], posicoes[0])
        if posicao >= 0
    ]

    resultados.sort(key=lambda resultado: resultado[0], reverse=True)
    selecionados = [item for _, item in resultados]

    contexto = "\n\n".join(f"[Página {item['pagina']}] {item['texto']}" for item in selecionados)
    paginas = sorted({item["pagina"] for item in selecionados})
    return contexto, paginas


def _montar_texto_reclamacao(row: dict) -> str:
    partes = [f"{k}: {str(v).strip()}" for k, v in row.items() if v and str(v).strip()]
    return "\n".join(partes) or "Reclamação sem texto identificado."


def _parse_json_result(texto: str) -> dict:
    if not texto:
        return {}
    
    texto_limpo = texto.strip()
    
    if "```" in texto_limpo:
        partes = texto_limpo.split("```")
        for parte in partes:
            parte_f = parte.strip()
            if parte_f.startswith("json"):
                parte_f = parte_f[4:].strip()
            if "{" in parte_f and "}" in parte_f:
                texto_limpo = parte_f
                break

    while texto_limpo.startswith("{{"):
        texto_limpo = "{" + texto_limpo[2:].lstrip()

    try:
        return json.loads(texto_limpo)
    except json.JSONDecodeError:
        pass

    try:
        inicio = texto_limpo.find("{")
        if inicio != -1:
            sub_json = texto_limpo[inicio:]
            if not sub_json.endswith("}"):
                sub_json = sub_json.rstrip()
                if not sub_json.endswith('"'):
                    sub_json += '"'
                if sub_json.count("{") > sub_json.count("}"):
                    sub_json += "}"
            return json.loads(sub_json)
    except json.JSONDecodeError:
        pass

    try:
        cat = re.search(r'"categoria"\s*:\s*"([^"]+)"', texto_limpo)
        prod = re.search(r'"produto"\s*:\s*"([^"]+)"', texto_limpo)
        sent = re.search(r'"sentimento"\s*:\s*"([^"]+)"', texto_limpo)
        urg = re.search(r'"urgencia"\s*:\s*"([^"]+)"', texto_limpo)
        res = re.search(r'"resumo"\s*:\s*"([^"]+)"', texto_limpo)
        
        if cat or res:
            return {
                "produto": prod.group(1) if prod else "Não Identificado",
                "categoria": cat.group(1) if cat else "Outros",
                "sentimento": sent.group(1) if sent else "Neutro",
                "urgencia": urg.group(1) if urg else "Média",
                "resumo": (res.group(1) if res else "Resumo parcial recuperado.") + "...",
            }
    except Exception:
        pass

    logging.warning("Falha definitiva ao interpretar JSON. Texto recebido: %s", texto[:200])
    return {
        "produto": "Não Identificado",
        "categoria": "Outros",
        "sentimento": "Neutro",
        "urgencia": "Média",
        "resumo": "Erro de leitura do JSON gerado pela IA.",
        "acao_sugerida": "Revisar logs do sistema.",
    }


def _chamar_gemini(prompt: str) -> str:
    if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY não configurada.")
    
    global _ULTIMA_CHAMADA_GEMINI
    decorrido = time.monotonic() - _ULTIMA_CHAMADA_GEMINI
    espera = GOOGLE_REQUEST_DELAY_SECONDS - decorrido
    if espera > 0:
        logging.info("Aguardando %.1fs antes da próxima chamada Gemini", espera)
        time.sleep(espera)

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    payload = {
        "model": OPENAI_MODEL, 
        "temperature": 0.1,
        "max_tokens": 1200,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }
    
    resposta = client.chat.completions.create(**payload)
    _ULTIMA_CHAMADA_GEMINI = time.monotonic()
    return resposta.choices[0].message.content or "{}"


def _classificar_reclamacao(
    texto_reclamacao: str, contexto_politicas: str, paginas_fonte: list[int] | None = None,
    definicao_classificador: str = "", modelo_preferido: str = "",
) -> dict:
    prompt = f"""
    DEFINIÇÃO DO CLASSIFICADOR:
    {definicao_classificador.strip() or "Analise a reclamação com base nas políticas."}
    
    CONTEXTO DAS POLÍTICAS (Recuperado via FAISS):
    {contexto_politicas}
    
    RECLAMAÇÃO:
    {texto_reclamacao}
    """

    try:
        inicio_chamada = time.perf_counter()
        resposta_texto = _chamar_gemini(prompt)
        _registrar_log_chamada("gemini", "Sucesso", inicio_chamada)

        dados = _parse_json_result(resposta_texto)
        dados["produto_identificado"] = _normalizar_produto(dados.get("produto", "Não Identificado"))
        dados["categoria"] = _normalizar_categoria(dados.get("categoria", "Outros"))
        dados["sentimento"] = _normalizar_sentimento(dados.get("sentimento", "Neutro"))
        dados["urgencia"] = _normalizar_urgencia(dados.get("urgencia", "Média"))
        
        # Higienização de quebras de linha brutas ou literais (\n)
        resumo_bruto = str(dados.get("resumo", "Sem resumo")).strip()
        resumo_limpo = resumo_bruto.replace("\\n", " ").replace("\n", " ")
        resumo_limpo = re.sub(r"\s+", " ", resumo_limpo).strip()
        dados["resumo"] = resumo_limpo

        dados["status_analise"] = "Concluída"
        dados["mensagem_analise"] = "Análise realizada com sucesso."
        dados["paginas_fonte"] = paginas_fonte if paginas_fonte else []
        return dados
    except Exception as e:
        status = _status_do_erro(e)
        _registrar_log_chamada("gemini", "Erro", time.perf_counter(), str(e))
        return _classificacao_padrao(status, f"{status} durante a análise.", "Tentar novamente.", paginas_fonte)


def classificar_reclamacoes_csv(
    csv_file, pdf_file, definicao_classificador: str = "", provedor_preferido: str = "auto",
    modelo_preferido: str = "", limite_reclamacoes: int | None = None,
) -> list:
    if csv_file is None: return []

    try:
        conteudo_csv = csv_file.read().decode("utf-8-sig", errors="replace")
        leitor = csv.DictReader(io.StringIO(conteudo_csv))
        linhas = list(leitor) if leitor.fieldnames else [{"texto": l.strip()} for l in conteudo_csv.splitlines() if l.strip()]

        if limite_reclamacoes is not None:
            linhas = linhas[:max(0, limite_reclamacoes)]

        indice_rag = _criar_indice_rag(pdf_file)

        resultados = []
        for indice, linha in enumerate(linhas, start=1):
            texto = _montar_texto_reclamacao(linha)
            
            contexto_politicas, paginas_fonte = _recuperar_contexto(indice_rag, texto)
            
            classificacao = _classificar_reclamacao(
                texto, contexto_politicas, paginas_fonte, definicao_classificador, modelo_preferido,
            )
            
            classificacao.update({
                "linha": indice,
                "id": linha.get("id", ""),
                "data_reclamacao": linha.get("data_reclamacao", ""),
                "canal": linha.get("canal", ""),
                "texto_reclamacao": str(linha.get("texto_reclamacao", texto)).strip(),
                "produto": linha.get("produto", ""),
                "status": linha.get("status", ""),
            })
            resultados.append(classificacao)

        return resultados
    except Exception as exc:
        logging.error("Erro ao classificar CSV: %s", exc)
        return []


def obter_modelo_embedding() -> str:
    return RAG_EMBEDDING_MODEL
