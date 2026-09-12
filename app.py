import csv
import hmac
import io
import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))
load_dotenv(BASE_DIR / ".env")

from generator import classificar_reclamacoes_csv, limpar_logs_chamadas, obter_logs_chamadas

st.set_page_config(
    page_title="Classificador de Reclamações",
    page_icon="📩",
    layout="wide",
)

st.markdown(
    """
    <style>
        .st-key-login_shell {
            max-width: 560px;
            margin: 8vh auto 0;
        }
        [data-testid="stSidebar"] {
            background-color: #0f172a;
        }
        .stButton > button {
            background-color: #2563eb;
            color: white;
            border: none;
            border-radius: 0.5rem;
        }
        .stButton > button:hover {
            background-color: #1d4ed8;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _exibir_login() -> None:
    with st.container(key="login_shell"):
        st.title("FinGuard")
        st.subheader("Acesso restrito")
        st.caption("Entre com suas credenciais para acessar a análise de reclamações.")

        usuario_configurado = os.getenv("FINGUARD_USERNAME")
        senha_configurada = os.getenv("FINGUARD_PASSWORD")
        if not usuario_configurado or not senha_configurada:
            st.error(
                "Credenciais não configuradas. Defina FINGUARD_USERNAME e "
                "FINGUARD_PASSWORD no arquivo .env."
            )
            return

        with st.form("login_form"):
            usuario = st.text_input("Usuário")
            senha = st.text_input("Senha", type="password")
            entrar = st.form_submit_button("Entrar", type="primary", width="stretch")

        if entrar:
            usuario_valido = hmac.compare_digest(usuario, usuario_configurado)
            senha_valida = hmac.compare_digest(senha, senha_configurada)
            if usuario_valido and senha_valida:
                st.session_state.autenticado = True
                st.rerun()
            st.error("Usuário ou senha inválidos.")


if not st.session_state.get("autenticado", False):
    _exibir_login()
    st.stop()

with st.sidebar:
    st.title("📥 Entrada")
    st.caption("Arquivo CSV com reclamações e PDF com políticas da empresa.")

    if st.button("Sair", width="stretch"):
        st.session_state.autenticado = False
        st.session_state.pop("resultado", None)
        st.rerun()

    modelos = {
        "Automático (Bedrock, depois Gemini)": ("auto", ""),
        "Bedrock: openai.gpt-oss-120b": ("openai-compatible", "openai.gpt-oss-120b"),
        "Google: gemini-flash-latest": ("gemini", "gemini-flash-latest"),
    }
    modelo_selecionado = st.selectbox("Modelo para esta execução", list(modelos))
    provedor_preferido, modelo_preferido = modelos[modelo_selecionado]

    csv_file = st.file_uploader("Selecione o CSV das reclamações", type=["csv"])
    pdf_file = st.file_uploader("Selecione o PDF de políticas", type=["pdf"])
    quantidade_opcoes = ["Todas"]
    if csv_file is not None:
        conteudo_preview = csv_file.getvalue().decode("utf-8-sig", errors="replace")
        total_reclamacoes = max(0, len(list(csv.DictReader(io.StringIO(conteudo_preview)))))
        quantidade_opcoes.extend(str(numero) for numero in range(1, total_reclamacoes + 1))
    quantidade_selecionada = st.selectbox(
        "Quantidade de reclamações",
        quantidade_opcoes,
        help="Escolha quantas linhas do CSV serão enviadas para classificação.",
    )
    definicao_classificador = st.text_area(
        "Instruções da análise FinGuard",
        value=(
            "Analise a reclamação usando as políticas do PDF. "
            "Explique a decisão de forma objetiva e indique uma ação prática."
        ),
        height=120,
    )

    if st.button("Classificar reclamações", type="primary", width="stretch"):
        if csv_file is None or pdf_file is None:
            st.warning("Envie ambos os arquivos: CSV e PDF de políticas.")
        else:
            with st.spinner("Lendo as políticas e classificando cada reclamação..."):
                limpar_logs_chamadas()
                resultado = classificar_reclamacoes_csv(
                    csv_file,
                    pdf_file,
                    definicao_classificador=definicao_classificador,
                    provedor_preferido=provedor_preferido,
                    modelo_preferido=modelo_preferido,
                    limite_reclamacoes=(
                        None if quantidade_selecionada == "Todas" else int(quantidade_selecionada)
                    ),
                )
                st.session_state.resultado = resultado
                st.success("Classificação concluída.")

st.title("📩 Classificador de Reclamações")
st.caption("Classifica registros de reclamação com base nas políticas do PDF enviado.")

if "resultado" in st.session_state and st.session_state.resultado:
    dados = st.session_state.resultado

    st.subheader("Resumo")
    total = len(dados)
    canais = {}
    for item in dados:
        canal = item["canal"] or "Não informado"
        canais[canal] = canais.get(canal, 0) + 1

    col1, col2 = st.columns(2)
    col1.metric("Total de reclamações", total)
    col2.metric("Canais detectados", len(canais))

    st.bar_chart({k: v for k, v in canais.items()}, horizontal=True)

    st.subheader("Detalhes da classificação")
    aba_dados, aba_analise, aba_logs = st.tabs(
        ["Dados da reclamação", "Análise FinGuard", "Logs das chamadas"]
    )

    with aba_dados:
        st.dataframe(
            [
                {
                    "ID": item["id"],
                    "Data da reclamação": item["data_reclamacao"],
                    "Canal": item["canal"],
                    "Texto da reclamação": item["texto_reclamacao"],
                    "Produto": item["produto"],
                    "Status": item["status"],
                }
                for item in dados
            ],
            width="stretch",
            hide_index=True,
        )

    with aba_analise:
        st.dataframe(
            [
                {
                    "ID": item["id"],
                    "Status da análise": item["status_analise"],
                    "Categoria": item["categoria"],
                    "Produto": item["produto_identificado"],
                    "Sentimento": item["sentimento"],
                    "Urgência": item["urgencia"],
                    "Resumo": item["resumo"],
                }
                for item in dados
            ],
            width="stretch",
            hide_index=True,
        )

    with aba_logs:
        logs = obter_logs_chamadas()
        if logs:
            st.dataframe(logs, width="stretch", hide_index=True)
        else:
            st.info("Nenhuma chamada registrada nesta execução.")

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
            "id",
            "data_reclamacao",
            "canal",
            "texto_reclamacao",
            "produto",
            "produto_identificado",
            "status",
            "categoria",
            "sentimento",
            "urgencia",
            "resumo",
            "prioridade",
            "criticidade",
            "fraude_identificada",
            "violacao_regulatoria",
            "escalonar_compliance",
            "area_responsavel",
            "status_analise",
            "mensagem_analise",
            "motivo",
            "acao_sugerida",
        ],
    )
    writer.writeheader()
    for item in dados:
        writer.writerow({
            "id": item["id"],
            "data_reclamacao": item["data_reclamacao"],
            "canal": item["canal"],
            "texto_reclamacao": item["texto_reclamacao"],
            "produto": item["produto"],
            "produto_identificado": item["produto_identificado"],
            "status": item["status"],
            "categoria": item["categoria"],
            "sentimento": item["sentimento"],
            "urgencia": item["urgencia"],
            "resumo": item["resumo"],
            "prioridade": item["prioridade"],
            "criticidade": item["criticidade"],
            "fraude_identificada": item["fraude_identificada"],
            "violacao_regulatoria": item["violacao_regulatoria"],
            "escalonar_compliance": item["escalonar_compliance"],
            "area_responsavel": item["area_responsavel"],
            "status_analise": item["status_analise"],
            "mensagem_analise": item["mensagem_analise"],
            "motivo": item["motivo"],
            "acao_sugerida": item["acao_sugerida"],
        })

    st.download_button(
        "Baixar resultados em CSV",
        data=csv_buffer.getvalue(),
        file_name="reclamacoes_classificadas.csv",
        mime="text/csv",
    )

else:
    st.info("Carregue um CSV e um PDF de políticas para começar a classificação.")
