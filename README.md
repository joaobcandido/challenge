# FinGuard

Assistente inteligente para triagem e análise de reclamações de clientes de instituições financeiras baseado em RAG (Retrieval-Augmented Generation).

O FinGuard processa um arquivo CSV contendo as reclamações dos clientes e um documento PDF com as políticas corporativas da instituição. A aplicação utiliza indexação semântica vetorial (FAISS + Embeddings) combinada com o Google Gemini para classificar registros com precisão, avaliar o sentimento, a urgência e gerar resumos gerenciais estruturados.

## Funcionalidades

- **Autenticação Segura:** Acesso restrito via login protegido por variáveis de ambiente (`.env`).
- **Indexação RAG Vetorial:** Processamento e busca semântica de políticas corporativas via FAISS e modelo de embedding `text-embedding-004`.
- **Upload Direto:** Envio intuitivo de arquivos CSV e PDF através da interface web.
- **Controle de Carga:** Seleção dinâmica da quantidade de registros do CSV a serem processados por execução.
- **Classificação Automatizada:** Triagem estruturada por Categoria, Produto, Sentimento e Urgência.
- **Resumo Analítico Fluido:** Geração de resumos gerenciais detalhados de 2 a 3 linhas, higienizados automaticamente contra quebras de linha indesejadas.
- **Auditoria de Chamadas:** Aba dedicada para monitoramento de latência, status, duração e logs operacionais da API.
- **Exportação de Dados:** Download imediato dos resultados consolidados em formato CSV.

## Fluxo da aplicação

1. O usuário realiza o login na tela de acesso restrito.
2. Faz o upload do arquivo CSV de reclamações e do PDF de políticas corporativas.
3. Define opcionalmente as instruções personalizadas para o classificador na barra lateral.
4. Escolhe se deseja processar todas as reclamações ou um limite específico de linhas.
5. A aplicação indexa o PDF, gera os vetores semânticos e executa a triagem via modelo de IA.
6. Os resultados consolidados são apresentados nas abas visuais (*Dados da reclamação* e *Análise FinGuard*).
7. O relatório completo pode ser exportado a qualquer momento via botão de download.

## Formato do CSV

O CSV de entrada deve conter obrigatoriamente estas colunas base:

```text
id,data_reclamacao,canal,texto_reclamacao,produto,status

Exemplo prático:

Snippet de código
id,data_reclamacao,canal,texto_reclamacao,produto,status
REC-2026-00001,2026-09-01,SAC,"Foi cobrado um valor duplicado no cartão.",Cartão,Em aberto
REC-2026-00002,2026-09-02,Ouvidoria,"Meu empréstimo foi calculado com juros incorretos.",Empréstimo,Em aberto
O campo essencial para a inteligência da análise é o texto_reclamacao. Os demais dados são preservados integralmente e exibidos no painel.

PDF de políticas
O documento de políticas serve como base de conhecimento RAG para orientar as decisões da IA, contendo diretrizes internas como:

Prazos de atendimento e resolução;

Regras de cobrança, cancelamento e reembolso;

Procedimentos de segurança e conformidade;

Critérios de criticidade e priorização.

Configuração local
Pré-requisitos
Python 3.11 ou superior;

pip;

Chave válida de acesso à API configurada.

Instalação
Clone o repositório e configure o ambiente virtual:

Bash
git clone [https://github.com/joaobcandido/challenge.git](https://github.com/joaobcandido/challenge.git)
cd challenge
python -m venv .venv
Ative o ambiente virtual conforme o seu sistema operacional:

Windows PowerShell:

PowerShell
.venv\Scripts\Activate.ps1
Git Bash no Windows:

Bash
source .venv/Scripts/activate
Linux/macOS:

Bash
source .venv/bin/activate
Instale as dependências necessárias:

Bash
python -m pip install -r requirements.txt
Variáveis de ambiente
Crie um arquivo .env na raiz do projeto com as suas credenciais. Não versione este arquivo:

Snippet de código
OPENAI_API_KEY=sua_credencial_de_api
OPENAI_BASE_URL=[https://generativelanguage.googleapis.com/v1beta/openai/](https://generativelanguage.googleapis.com/v1beta/openai/)
OPENAI_MODEL=gemini-1.5-flash-latest
RAG_EMBEDDING_MODEL=text-embedding-004
GOOGLE_REQUEST_DELAY_SECONDS=12
FINGUARD_USERNAME=seu_usuario
FINGUARD_PASSWORD=sua_senha_forte
Nota de segurança: Nunca exponha chaves de API reais no README, em repositórios públicos ou em capturas de tela.

Executar a aplicação
Inicie o servidor de desenvolvimento do Streamlit:

Bash
streamlit run app.py
Acesse a interface no navegador em http://localhost:8501.

Executar com Docker
Para rodar a aplicação em um ambiente conteinerizado:

Bash
docker compose up --build
Acesse http://localhost:8501. O arquivo docker-compose.yaml gerencia de forma integrada o carregamento das variáveis do arquivo .env.

Estrutura principal
Plaintext
app.py                    # Interface Streamlit, roteamento de abas e autenticação
src/generator.py          # Lógica RAG (FAISS), prompts, parsers e integração com a IA
data/reclamacoes_exemplo.csv
data/pdfs/                # Repositório para PDFs de referência
requirements.txt          # Dependências do projeto Python
Dockerfile
docker-compose.yaml
Validação rápida de sintaxe
Bash
python -m py_compile app.py src/generator.py
