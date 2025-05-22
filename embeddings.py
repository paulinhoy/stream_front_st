import pickle
from sentence_transformers import SentenceTransformer
import pdfplumber

# Função para extrair texto dos PDFs
def extract_documents(pdf_paths):
    documents = []
    current_id = 1
    for path in pdf_paths:
        with pdfplumber.open(path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    documents.append({
                        'pdf': path,
                        'id': current_id,
                        'page_number': page_number,
                        'text': text
                    })
                    current_id += 1
    return documents

# Caminhos dos PDFs
pdf_paths = ['mecanica.pdf', 'Falhas.pdf', 'eletrica.pdf']

# Carregar modelo (pode ser o "paraphrase-MiniLM-L3-v2" para mais velocidade)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extrair documentos
documents = extract_documents(pdf_paths)

# Calcular embeddings
doc_embeddings = {}
for doc in documents:
    doc_embeddings[doc["id"]] = model.encode(doc['text'])

# Salvar embeddings e documentos
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(doc_embeddings, f)

with open('documents.pkl', 'wb') as f:
    pickle.dump(documents, f)

print("Embeddings e documentos salvos!")