import streamlit as st
import base64
from io import BytesIO
import os
import pdfplumber
import re
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz
import requests
import pickle

# Carregar chave da OpenAI dos secrets do Streamlit Cloud
api_key = st.secrets["OPENAI_API_KEY"]

# Carregar modelo para gerar embedding da query
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
model = load_model()

# Carregar documentos e embeddings já salvos (pré-processados offline)
@st.cache_resource
def load_data():
    with open('documents.pkl', 'rb') as f:
        documents = pickle.load(f)
    with open('embeddings.pkl', 'rb') as f:
        doc_embeddings = pickle.load(f)
    return documents, doc_embeddings

documents, doc_embeddings = load_data()

# Função para extrair imagem da página do PDF (pode ser cacheada)
@st.cache_data
def get_page_image_base64(pdf_path, page_number):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            pil_image = page.to_image(resolution=100).original
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str
    except Exception as e:
        print(f"Erro ao extrair imagem: {e}")
        return None

def carregar_falhas_txt(caminho="falhas.txt"):
    erros_paginas = {}
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if not linha:
                    continue
                codigo, paginas = linha.split()
                lista_paginas = [int(p) for p in paginas.split(",")]
                erros_paginas[codigo] = lista_paginas
    except Exception as e:
        print(f"Erro ao carregar falhas.txt: {e}")
    return erros_paginas

def extrair_codigo_erro(texto, lista_codigos):
    match = re.search(r"s?r?v?o[\s\-\/_]?(\d{1,3})", texto, re.IGNORECASE)
    if match:
        numero = int(match.group(1))
        return f"SRVO-{numero:03d}"
    palavras = re.findall(r'\w+', texto)
    melhor_codigo, melhor_score = None, 0
    for palavra in palavras:
        resultado = process.extractOne(
            palavra.upper(), lista_codigos, scorer=fuzz.ratio
        )
        if resultado and resultado[1] > melhor_score and resultado[1] > 80:
            melhor_codigo, melhor_score = resultado[0], resultado[1]
    return melhor_codigo

def extract_pdf_and_page(text):
    match = re.search(r'Fonte:\s*([^\s,]+),\s*p[áa]gina\s*(\d+)', text, re.IGNORECASE)
    if match:
        pdf = match.group(1)
        page = int(match.group(2))
        return pdf, page
    return None, None

# Carregar falhas e códigos
erros_paginas = carregar_falhas_txt()
lista_codigos = list(erros_paginas.keys())

# Interface Streamlit
st.markdown("""
    <style>
    .title {
    text-align: center;
    padding: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("<h1 class='title'>Chat - Stihl</h1>", unsafe_allow_html=True)

query = st.text_input("Digite sua pergunta:")

if st.button("Perguntar") and query:
    with st.spinner("Consultando..."):
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = []
        codigo_erro = extrair_codigo_erro(query, lista_codigos)
        for doc in documents:
            score = util.cos_sim(query_embedding, doc_embeddings[doc["id"]]).item()
            scores.append((score, doc))
        top_docs = sorted(scores, key=lambda x: x[0], reverse=True)[:15]  
        context = "\n".join([
            f"Documento {doc['pdf']}, página {doc['page_number']}: {doc['text']}"
            for _, doc in top_docs
        ])

        contexto_falha = ""
        if codigo_erro and codigo_erro in erros_paginas:
            paginas = erros_paginas[codigo_erro]
            trechos = []
            for doc in documents:
                if doc['pdf'].lower() == "falhas.pdf" and doc['page_number'] in paginas:
                    trechos.append(
                        f"Documento {doc['pdf']}, página {doc['page_number']}: {doc['text']}"
                    )
            if trechos:
                contexto_falha = (
                    f"Informações diretamente relacionadas ao código {codigo_erro}:\n"
                    + "\n".join(trechos)
                    + "\n\n"
                )

        prompt_system = (
            "Você é um assistente de IA. Responda a pergunta do usuário com base apenas nos documentos fornecidos. "
            "Ao final da sua resposta, SEMPRE informe a fonte no seguinte formato, em uma linha separada: "
            "Se foi utilizado mais de uma referência para a construção da respota informe apenas a fonte mais importante.  "
            "Fonte: <nome_do_pdf>, página <número_da_página> "
            "Exemplo: Fonte: mecanica.pdf, página 12"
        )
        prompt_user = f"Documentos:\n{contexto_falha}{context}\n\nPergunta: {query}"

        best_doc = top_docs[0][1]

        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user}
                ]
            }
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            res = response.json()
            if "choices" in res and len(res["choices"]) > 0:
                resposta = res["choices"][0]["message"]["content"]
                pdf_citado, pagina_citada = extract_pdf_and_page(resposta)
                if pdf_citado and pagina_citada:
                    image_base64 = get_page_image_base64(pdf_citado, pagina_citada)
                else:
                    image_base64 = get_page_image_base64(best_doc['pdf'], best_doc['page_number'])
                st.text_area("Resposta do Assistente", resposta, height=200)
                if image_base64:
                    img_bytes = base64.b64decode(image_base64)
                    st.image(BytesIO(img_bytes), caption=f"PDF: {pdf_citado or best_doc['pdf']}, Página: {pagina_citada or best_doc['page_number']}")
                else:
                    st.info("Nenhuma imagem retornada.")
            else:
                st.error("Resposta inesperada da OpenAI.")
        except Exception as e:
            st.error(f"Erro: {e}")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.8em;'>Desenvolvido pela equipe Vent</p>",
    unsafe_allow_html=True
)