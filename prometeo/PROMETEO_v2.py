import ast
import numpy as np
import os
import pandas as pd
import textwrap
import tiktoken
import umap.umap_ as umap
from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
from typing import Optional, List, Dict

# Funciones de utilidad
def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))

def reduce_cluster_embeddings(embeddings: np.ndarray, dim: int, n_neighbors: Optional[int] = None, metric: str = "cosine") -> np.ndarray:
    n_neighbors = n_neighbors or int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)

def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 1234) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    bics = [GaussianMixture(n_components=n, random_state=random_state).fit(embeddings).bic(embeddings) for n in range(1, max_clusters)]
    return np.argmin(bics) + 1

def gmm_clustering(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters

def format_cluster_texts(df: pd.DataFrame) -> Dict[int, str]:
    return {cluster: " --- ".join(df[df['Cluster'] == cluster]['Texto'].tolist()) for cluster in df['Cluster'].unique()}

def wrap_text_preserve_newlines(text: str, width: int = 80) -> str:
    return '\n'.join([textwrap.fill(line, width=width) for line in text.split('\n')])

def process_llm_response(llm_response: Dict):
    print(wrap_text_preserve_newlines(llm_response['answer']))
    print('\nReferencias:')
    for contexto in llm_response["context"][:5]:
        print(contexto)
    print('\n\n')

# Configuración inicial
load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()
detailed_llm = ChatOpenAI(temperature=0, model_name='gpt-4o-mini')
llm = ChatOpenAI(temperature=0.5, model_name='gpt-4o-mini')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)

# Funciones principales
def create_conversation_theme():
    docs_dir = './docs/'
    if not os.path.exists(docs_dir):
        print(f"Error: El directorio '{docs_dir}' no existe.")
        print("Por favor, crea este directorio y coloca tus archivos PDF en él.")
        return None, None, None

    pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"Error: No se encontraron archivos PDF en el directorio '{docs_dir}'.")
        print("Por favor, asegúrate de colocar al menos un archivo PDF en este directorio.")
        return None, None, None

    documents = DirectoryLoader(docs_dir, glob="./*.pdf", loader_cls=PyPDFLoader).load()
    for d in documents:
        d.page_content = d.page_content.replace('\n', ' ').replace('\t', ' ')
    
    docs = text_splitter.split_documents(documents)
    texts = [doc.page_content for doc in docs]
    
    concatenated_content = "\n\n\n --- \n\n\n".join([doc.page_content for doc in reversed(sorted(docs, key=lambda x: x.metadata["source"]))])
    print(f"Número de tokens en el documento proporcionado: {num_tokens_from_string(concatenated_content)}")
    
    global_embeddings = [embeddings.embed_query(txt) for txt in texts]
    
    topic_name = input('¿Cómo se llama el tema de conversación? ')
    embed_name = f'{topic_name}_emb.txt'
    
    with open(embed_name, 'w') as f:
        for i in global_embeddings:
            f.write("%s\n" % i)
    print(f'\nEstás usando el tema de conversación: {embed_name}')
    
    # Resto del proceso de creación...
    
    return topic_name, global_embeddings, texts

def load_conversation_theme():
    topic_name = input('¿Cómo se llama el tema de conversación? ')
    embed_name = f'{topic_name}_emb.txt'
    print(f'Estás usando el tema de conversación: {embed_name}\n')
    
    with open(embed_name, 'r') as f:
        global_embeddings = [ast.literal_eval(i.strip()) for i in f]
    
    global_embeddings = np.array(global_embeddings, dtype=float)
    
    with open(f'{topic_name}.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    textos = text_splitter.split_text(content)
    
    return topic_name, global_embeddings, textos

def setup_vectorstore(topic_name: str, texts: List[str]):
    persist_directory = f'{topic_name}_kb'
    vectorstore = Chroma.from_texts(texts=texts, embedding=embeddings, persist_directory=persist_directory)
    vectorstore.persist()
    return vectorstore

def adjust_final_number(vectorstore: Chroma, string: str, max_threshold: int, initial_number: int) -> int:
    final_number = initial_number
    while final_number < max_threshold:
        retriever = vectorstore.as_retriever(search_kwargs={"k": final_number})
        docs = retriever.invoke(string)
        text = "".join([doc.page_content for doc in docs])
        if num_tokens_from_string(text) < max_threshold:
            final_number += 1
        else:
            break
    return final_number

def setup_rag_chain(retriever, llm):
    template = """
    Eres Prometeo, un asistente especializado en revisión bibliográfica, que habla Español.

    Tu tarea consiste en proporcionar respuestas extremadamente detalladas y basadas en evidencia a 
    cualquier pregunta relacionada con el siguiente contexto, obtenido de un artículo científico: {context}.

    Tu respuesta debe centrarse en los aspectos más relevantes de la literatura científica sin mencionar 
    directamente el contexto proporcionado. Identifica y utiliza palabras clave para enfocarte en los temas 
    más importantes para dar una respuesta más precisa.

    Siempre responde en Español, excepto en nombres propios, y no menciones detalles sobre ti a menos que se 
    te pregunte directamente.

    Finalmente y teniendo en cuento lo anterior, responde la siguiente pregunta: {question}
    """

    selected_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: "\n\n".join(doc.page_content for doc in x["context"])))
        | selected_prompt
        | llm
        | StrOutputParser()
    )

    return RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain)

# Flujo principal
def main():
    user_input = input("¿Crear (1) o cargar (2) un tema de conversación?: ")

    if user_input == "1":
        topic_name, global_embeddings, texts = create_conversation_theme()
    elif user_input == "2":
        topic_name, global_embeddings, texts = load_conversation_theme()
    else:
        print('No seleccionaste ningún tema de conversación.\n')
        return

    vectorstore = setup_vectorstore(topic_name, texts)
    final_number = adjust_final_number(vectorstore, "¿Cuál es el tema principal del documento?", 10000, 4)
    print(f'K final es: {final_number}')
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": final_number})
    rag_chain_with_source = setup_rag_chain(retriever, llm)

    query = input("Hazme una pregunta: ")
    print(query + '\n')
    llm_response = rag_chain_with_source.invoke(query)
    process_llm_response(llm_response)

if __name__ == "__main__":
    main()