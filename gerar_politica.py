from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors

def criar_pdf_politica():
    nome_arquivo = "politica_empresa.pdf"
    doc = SimpleDocTemplate(
        nome_arquivo, 
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Estilos customizados
    titulo_style = ParagraphStyle(
        'TituloDoc',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=15
    )
    
    regra_titulo_style = ParagraphStyle(
        'RegraTitulo',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=4
    )
    
    corpo_style = ParagraphStyle(
        'CorpoTexto',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748")
    )

    elements = []

    # Cabeçalho do Documento
    elements.append(Paragraph("<b>Política Interna de Atendimento ao Cliente</b>", titulo_style))
    elements.append(Paragraph("Diretrizes oficiais para tratamento de solicitações, reclamações e suporte.", corpo_style))
    elements.append(Spacer(1, 15))

    # As 5 Regras da Empresa
    regras = [
        ("1. Prazo Máximo de Resposta ao Cliente", 
         "Todas as reclamações e solicitações recebidas pelos canais de atendimento devem ter uma resposta inicial em até <b>24 horas úteis</b>."),
        
        ("2. Transparência na Resolução de Erros", 
         "Caso o problema seja decorrente de falha operacional ou sistêmica da empresa, o cliente deve ser informado com clareza sobre a causa e a previsão exata de correção."),
        
        ("3. Política de Reembolso e Estornos", 
         "Cobranças duplicadas ou indevidas têm garantia de devolução integral. O processo de reembolso deve ser iniciado em no máximo <b>3 dias úteis</b> após a validação do departamento financeiro."),
        
        ("4. Troca de Produtos Danificados ou Incorretos", 
         "Produtos entregues com avarias, defeitos de fabricação ou especificações incorretas serão trocados sem nenhum custo adicional de frete para o consumidor."),
        
        ("5. Confidencialidade e Segurança de Dados", 
         "É estritamente proibido o compartilhamento de dados pessoais ou financeiros de clientes com terceiros não autorizados, em conformidade com as diretrizes da LGPD.")
    ]

    for titulo, descricao in regras:
        elements.append(Paragraph(titulo, regra_titulo_style))
        elements.append(Paragraph(descricao, corpo_style))
        elements.append(Spacer(1, 8))

    doc.build(elements)
    print(f"PDF '{nome_arquivo}' gerado com sucesso!")

if __name__ == "__main__":
    criar_pdf_politica()
