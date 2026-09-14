import os
import zipfile
from pathlib import Path

# Nome do arquivo ZIP de saída
NOME_ZIP = "finguard_project.zip"

# Pastas e arquivos que devem ser ignorados na compactação
EXCLUIR_PASTAS = {
    ".venv",
    "__pycache__",
    ".git",
    "rag_index",
}

EXCLUIR_ARQUIVOS = {
    ".env",
    "finguard_project.zip",
}

def criar_zip():
    diretorio_raiz = Path(__file__).resolve().parent
    arquivo_destino = diretorio_raiz / NOME_ZIP
    
    print(f"Compactando o projeto em '{NOME_ZIP}' (bloqueando .env, mantendo .env-example)...")
    
    contador = 0
    with zipfile.ZipFile(arquivo_destino, "w", zipfile.ZIP_DEFLATED) as zipf:
        for caminho_atual, subpastas, arquivos in os.walk(diretorio_raiz):
            # Remove pastas indesejadas para o os.walk não entrar nelas
            subpastas[:] = [d for d in subpastas if d not in EXCLUIR_PASTAS and not d.endswith(".egg-info")]
            
            caminho_atual_obj = Path(caminho_atual)
            
            # Pula caso o caminho atual faça parte de uma pasta excluída
            if any(parte in EXCLUIR_PASTAS for parte in caminho_atual_obj.parts):
                continue
                
            for arquivo in arquivos:
                # Bloqueia apenas o .env exato e o próprio arquivo zip gerado (permite .env-example)
                if arquivo in EXCLUIR_ARQUIVOS:
                    continue
                    
                arquivo_path = caminho_atual_obj / arquivo
                
                # Calcula o caminho relativo para manter a estrutura correta dentro do zip
                relpath = arquivo_path.relative_to(diretorio_raiz)
                
                zipf.write(arquivo_path, relpath)
                contador += 1
                print(f"  Adicionado: {relpath}")

    print(f"\nSucesso! {contador} arquivos compactados com segurança no arquivo '{NOME_ZIP}'.")

if __name__ == "__main__":
    criar_zip()
