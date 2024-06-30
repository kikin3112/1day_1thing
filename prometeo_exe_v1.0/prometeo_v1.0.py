# Paquetería para todo el heavy-lifting

import ast
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import textwrap
import tiktoken
import umap.umap_ as umap
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
from dotenv import load_dotenv, find_dotenv
from typing import Optional

##############
### PARAMS ###
##############

# Se especifica el servicio de embeddings a usar
load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings(
)

# Se especifíca el LLM para: "Resumen extremadamente
# detallado"
detailed_turbo_llm = turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo-0125'
)

template = """Tu tarea como revisor bibliográfico profesional
es crear resúmenes extremadamente detallados del siguiente
texto: {text} """

prompt = PromptTemplate.from_template(template)
chain = prompt | detailed_turbo_llm | StrOutputParser()

# Se especifíca el LLM para: "Ser Prometeo"
turbo_llm = ChatOpenAI(
    temperature=0.5,
    model_name='gpt-3.5-turbo-0125'
)

# Se especifíca el porcionador de tokens
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100
)

####################
### DOC-HANDLING ###
####################

# Carga de documentos y extracción de información
# (asegúrate de que haya PDFs en la carpeta documentos)
documents = DirectoryLoader('./docs/', glob="./*.pdf", loader_cls=PyPDFLoader).load()
# Tratameinto de caracteres indeseados
for d in documents:
    d.page_content = d.page_content.replace('\n', ' ').replace('\t', ' ')

docs = text_splitter.split_documents(documents)
texts = [doc.page_content for doc in docs]


def num_tokens_from_string(string: str) -> int:
    """Esta función se encarga de contar el numero
    total de tokens en el documento proporcionado"""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)
print(
    "Número total de tokens en el documento: %s"
    % num_tokens_from_string(concatenated_content)
)
print('-------')
print('')
#############################
### CREATE-LOAD EMBEDDING ###
#############################

user_input_emb = input("¿Crear (1) o cargar (2) una incrustación?: ")

destino_emb = r'c:\Users\luisr\OneDrive\Documentos\GitHub\1day_1thing\prometeo\embed'

if user_input_emb.lower() == "1":
    print('Elegiste 1')
    print('-------')
    print('')
    global_embeddings = [embeddings.embed_query(txt) for txt in texts]

    embed_name = input('¿Cómo se llama esta incrustación?: ') + '_emb' + '.txt'
    emb = rf'c:\Users\luisr\OneDrive\Documentos\GitHub\1day_1thing\prometeo\{embed_name}'
    with open(rf'./{embed_name}', 'w') as f:
        for i in global_embeddings:
            f.write("%s\n" % i)
    shutil.move(emb, destino_emb)
    
elif user_input_emb.lower() == "2":
    print('Elegiste 2')
    print('-------')
    print('')
    global_embeddings = []

    embed_name = input('Nombre de la incrustación: ') + '_emb' + '.txt'

    with open(rf'./embed/{embed_name}', 'r') as f:
        for i in f:
            x = ast.literal_eval(i.strip())  # Convertir la cadena a lista de números
            global_embeddings.append(x)

    global_embeddings = np.array(global_embeddings, dtype=float)
    
elif user_input_emb != "1" and user_input_emb != "2":
    print("No seleccionaste ninguna incrustación.")

#########################################
### ALGO DIM-REDUCTION FOR CLUSTERING ###
#########################################

def reduce_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """Explicar el algoritmo de reducción de dimensiones"""
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)

def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 1234):
    max_clusters = min(max_clusters, len(embeddings))
    bics = [GaussianMixture(n_components=n, random_state=random_state).fit(embeddings).bic(embeddings)
            for n in range(1, max_clusters)]
    return np.argmin(bics) + 1

def gmm_clustering(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters

def format_cluster_texts(df):
    clustered_texts = {}
    for cluster in df['Cluster'].unique():
        cluster_texts = df[df['Cluster'] == cluster]['Texto'].tolist()
        clustered_texts[cluster] = " --- ".join(cluster_texts)
    return clustered_texts

dim = 2
global_embeddings_reduced = reduce_cluster_embeddings(global_embeddings, dim)
labels, _ = gmm_clustering(global_embeddings_reduced, threshold=0.5)
simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]

########################
### DF FOR SUMMARIES ###
########################

df = pd.DataFrame({
    'Texto': texts,
    'Embedding': list(global_embeddings_reduced),
    'Cluster': simple_labels
})

clustered_texts = format_cluster_texts(df)
summaries = {}
for cluster, text in clustered_texts.items():
    summary = chain.invoke({"text": text})
    summaries[cluster] = summary

embedded_summaries = [embeddings.embed_query(summary) for summary in summaries.values()]
embedded_summaries_np = np.array(embedded_summaries)
labels, _ = gmm_clustering(embedded_summaries_np, threshold=0.5)
simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]

clustered_summaries = {}
for i, label in enumerate(simple_labels):
    if label not in clustered_summaries:
        clustered_summaries[label] = []
    clustered_summaries[label].append(list(summaries.values())[i])

final_summaries = {}
for cluster, texts in clustered_summaries.items():
    combined_text = ' '.join(texts)
    summary = chain.invoke({"text": combined_text})
    final_summaries[cluster] = summary

texts_from_df = df['Texto'].tolist()
texts_from_clustered_texts = list(clustered_texts.values())
texts_from_final_summaries = list(final_summaries.values())
combined_texts = texts_from_df + texts_from_clustered_texts + texts_from_final_summaries

#####################
### FINAL CONTEXT ###
#####################

file_name = input('Nombre del archivo: ') + '.txt'

# Escribir la lista en el archivo de texto
with open(file_name, 'w', encoding='utf-8') as f:
    for t in combined_texts:
        f.write("%s\n" % t)

# Leer el contenido del archivo y mostrarlo
with open(file_name, 'r', encoding='utf-8') as f:
    content = f.read()

textos = text_splitter.split_text(content)

###############################
### KNOWLEDGE-BASE CREATION ###
###############################

user_input_kb = input("¿Enseñar (1) o recordar (2) una knowledge-base?: ")

destino_kb = r'c:\Users\luisr\OneDrive\Documentos\GitHub\1day_1thing\prometeo\kbs'

if user_input_kb.lower() == "1":
    persist_directory = input('Nombre esta knowledge-base: ') + '_kb'
    vectorstore = Chroma.from_texts(texts=textos,
                                    embedding=embeddings,
                                    persist_directory=persist_directory)
    vectorstore.persist()
    vectorstore = None
    os.system(f'zip -r db.zip ./{persist_directory}')

    kb = rf'c:\Users\luisr\OneDrive\Documentos\GitHub\1day_1thing\prometeo\{persist_directory}'
    shutil.move(kb, destino_kb)