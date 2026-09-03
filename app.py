import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.retriever import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults

# 1. Configuração Inicial
st.set_page_config(page_title="Assistente de Logística e Transportes", page_icon="🚛")
st.title("🚛 Assistente de Legislação Rodoviária")

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", "")

# 2. Configurar a Base de Dados (RAG + Híbrida)
@st.cache_resource(show_spinner="A processar legislação da pasta...")
def preparar_base_dados():
    if not os.path.exists("legislacao"):
        os.makedirs("legislacao")
    loader = PyPDFDirectoryLoader("legislacao")
    docs = loader.load()
    
    if not docs:
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\nArtigo", "\nSECÇÃO", "\nCAPÍTULO", "\n\n", "\n", " ", ""],
        chunk_size=1200, chunk_overlap=150, length_function=len
    )
    splits = text_splitter.split_documents(docs)
    
    # Base Vetorial e Base de Palavras-chave
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3
    
    # Combinar as duas abordagens para não perder dados importantes como coimas
    hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
    return hybrid_retriever

retriever = preparar_base_dados()

# 3. Modelos e Ferramentas
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
web_search = DuckDuckGoSearchResults()

# 4. Prompt Engineering Avançado (O "Cérebro" das Regras)
system_prompt = """És um assistente jurídico especializado em logística de transportes de pesados em Portugal.
Tens duas fontes de informação. A tua missão é analisá-las nesta ordem ESTRITA de prioridade:

1. CONTEXTO DOS DOCUMENTOS INTERNOS (A tua prioridade máxima):
{context}

2. PESQUISA NA WEB (Usa apenas se os documentos internos não responderem à pergunta de forma completa):
{web_results}

REGRAS DE RESPOSTA OBRIGATÓRIAS:
- Dá SEMPRE prioridade aos documentos internos (RAG). Se a resposta estiver nos documentos, usa essa informação e ignora a web. Cita o nome do documento (ex: "Segundo o Artigo X do documento Y...").
- Se a resposta NÃO estiver nos documentos, então podes usar os dados da PESQUISA NA WEB. 
- Quando usares a pesquisa na web, inicia a resposta com: "Não encontrei esta informação na legislação carregada, mas com base na pesquisa online nos sites oficiais..."
- Dá sempre preferência e destaque às informações que apresentem a data mais recente.
- Se nenhuma das fontes tiver a resposta, diz que não sabes. Não tentes adivinhar a lei.
"""
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

# 5. Interface
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if pergunta := st.chat_input("Ex: O que mudou nos tacógrafos com a nova lei?"):
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)
        
    with st.chat_message("assistant"):
        with st.spinner("A analisar documentos e a consultar fontes oficiais..."):
            
            # Passo A: Pesquisar nos PDFs locais
            contexto_docs = ""
            if retriever:
                docs_recuperados = retriever.invoke(pergunta)
                contexto_docs = "\n\n".join([d.page_content for d in docs_recuperados])
            
            # Passo B: Fazer pesquisa web filtrada apenas para os sites exigidos
            query_pesquisa = f"{pergunta} site:act.gov.pt OR site:imt-ip.pt OR site:eur-lex.europa.eu OR site:trace.pt"
            resultados_web = web_search.run(query_pesquisa)
            
            # Passo C: Enviar tudo para o LLM processar as regras
            cadeia = prompt | llm
            resposta = cadeia.invoke({
                "context": contexto_docs if contexto_docs else "Nenhum documento interno disponível.",
                "web_results": resultados_web if resultados_web else "Sem resultados na web.",
                "input": pergunta
            })
            
            st.markdown(resposta.content)
            st.session_state.mensagens.append({"role": "assistant", "content": resposta.content})