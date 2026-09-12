# FinGuard

Assistente inteligente para triagem e análise de reclamações de clientes de instituições financeiras.

O FinGuard recebe um CSV de reclamações e um PDF com as políticas da empresa. A aplicação usa um modelo de linguagem para classificar cada registro, avaliar riscos e sugerir o próximo encaminhamento.

## Funcionalidades

- Upload de CSV e PDF diretamente pela interface;
- Seleção da quantidade de reclamações a processar;
- Seleção do provedor/modelo por execução;
- Classificação por categoria, produto, sentimento e urgência;
- Resumo padronizado de 2 a 3 linhas para cada reclamação;
- Identificação de indícios de fraude e violação regulatória;
- Indicação de escalonamento para Compliance e área responsável;
- Sugestão de ação para atendimento;
- Abas separadas para dados da reclamação, análise FinGuard e logs das chamadas;
- Exportação dos resultados para CSV;
- Login com usuário e senha configurados por variáveis de ambiente;
- Fallback opcional para Google Gemini;

## Fluxo da aplicação

1. O usuário acessa a página de login.
2. Seleciona o modelo, o CSV e o PDF de políticas.
3. Define, opcionalmente, as instruções da análise FinGuard.
4. Escolhe quantas reclamações deseja processar.
5. A aplicação envia cada reclamação junto com as políticas para o provedor selecionado.
6. Os resultados aparecem nas abas de dados e análise.
7. Os detalhes completos podem ser baixados em CSV.

## Formato do CSV

O CSV deve conter estas colunas:

```text
id,data_reclamacao,canal,texto_reclamacao,produto,status
```

Exemplo:

```csv
id,data_reclamacao,canal,texto_reclamacao,produto,status
REC-2026-00001,2026-09-01,SAC,"Foi cobrado um valor duplicado no cartão.",Cartão,Em aberto
REC-2026-00002,2026-09-02,Ouvidoria,"Meu empréstimo foi calculado com juros incorretos.",Empréstimo,Em aberto
```

O campo mais importante para a análise é `texto_reclamacao`. Os demais campos são preservados e exibidos separadamente.

## PDF de políticas

O nome do arquivo não precisa seguir um padrão. O PDF deve conter as regras usadas para orientar a análise, como:

- Prazos de atendimento;
- Regras de cobrança, cancelamento e reembolso;
- Procedimentos para fraude;
- Regras de privacidade;
- Critérios de escalonamento para Compliance;
- Responsabilidades das áreas internas.

## Modelos disponíveis

Na barra lateral, é possível selecionar:

- **Automático:** tenta o endpoint OpenAI-compatible e usa o Gemini como fallback;
- **Bedrock:** usa `openai.gpt-oss-120b` no endpoint configurado;
- **Google:** usa `gemini-flash-latest`.

O sistema registra o provedor, modelo, status, duração e detalhes de erro na aba **Logs das chamadas**. Chaves e prompts não são exibidos nos logs.

## Configuração local

### Pré-requisitos

- Python 3.11 ou superior;
- pip;
- Uma chave válida do provedor escolhido.

### Instalação

```bash
git clone https://github.com/joaobcandido/challenge.git
cd challenge
python -m venv .venv
```

Ative o ambiente virtual:

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Git Bash no Windows:

```bash
source .venv/Scripts/activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

### Variáveis de ambiente

Crie um arquivo `.env` na raiz. Não versione esse arquivo:

```env
OPENAI_API_KEY=sua_credencial_do_endpoint
OPENAI_MODEL=openai.gpt-oss-120b
OPENAI_BASE_URL=https://bedrock-mantle.us-east-1.api.aws/v1
GOOGLE_API_KEY=sua_chave_google
GOOGLE_REQUEST_DELAY_SECONDS=25
FINGUARD_USERNAME=seu_usuario
FINGUARD_PASSWORD=sua_senha_forte
```

Para usar apenas o Gemini, remova ou deixe vazio `OPENAI_BASE_URL` e `OPENAI_API_KEY`. A aplicação usará `GOOGLE_API_KEY`.

As credenciais devem ser revogadas e geradas novamente caso tenham sido expostas. Nunca coloque chaves reais no README, no GitHub ou em screenshots.

### Executar

```bash
streamlit run app.py
```

Acesse [http://localhost:8501](http://localhost:8501).

## Executar com Docker

```bash
docker compose up --build
```

Acesse [http://localhost:8501](http://localhost:8501). O `docker-compose.yaml` carrega as variáveis do arquivo `.env`.


## Estrutura principal

```text
app.py                    # Interface Streamlit e autenticação
src/generator.py          # Provedores, prompts e classificação
data/reclamacoes_exemplo.csv
data/pdfs/                # PDFs de exemplo
requirements.txt          # Dependências Python
Dockerfile
docker-compose.yaml
```

## Validação rápida

```bash
python -m py_compile app.py src/generator.py
```
