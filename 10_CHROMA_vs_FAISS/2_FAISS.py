"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro


FAISS  --> https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/
=====

Facebook AI Similarity Search é uma biblioteca para pesquisa de similaridade eficiente e agrupamento (clustering) 
de vetores densos. Contém algoritmos que buscam em conjuntos de vetores de qualquer tamanho, até aqueles que 
possivelmente não cabem na memória RAM. Ele também contém código de suporte para avaliação e ajuste de parâmetros.
"""
# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 

from langchain.chains import RetrievalQA # Tem que atualizar --> pip install langchai==0.0.137
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Carregando meu documento em PDF:
loader = PyPDFLoader("./Eddy_Running_the_Vale_API.pdf")
#documents = loader.load()
documents = loader.load_and_split()

print(documents)
print("")
print(len(documents))


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=0)

docs = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(docs, embeddings)


"""
No FAISS percebi que não aceita a DISTÂNCIA cosseno, nenhuma!
"""
query = "O que é system prune?"
docs_and_scores = db.similarity_search_with_score(query=query, k=6)

print(docs_and_scores)
print("")
print(docs_and_scores[0])
