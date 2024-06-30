# Paquetería para todo el heavy-lifting

import ast
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

##################
### PARÁMETROS ###
##################

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
    "Número total de tokens del docuemnto: %s"
    % num_tokens_from_string(concatenated_content)
)

