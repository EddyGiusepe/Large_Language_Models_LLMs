"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro


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


# Carregando meu documento em PDF:
loader = PyPDFLoader("./Eddy_Running_the_Vale_API.pdf")
documents = loader.load()

print(documents)
print("")
print(len(documents))


# Dividir os documentos em chunks:
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


# Selecione que EMBEDDINGS quer usar:
embeddings = OpenAIEmbeddings()

# Crie o vectorestore para usar como índice (index):
db = Chroma.from_documents(texts, embeddings)

""""
Lembrar que o SCORE na hora de realizar a Pesquisa por Similaridade é um float que está
relacionado com a Distância. Default é DISTÂNCIA EUCLIDIANA 🧐
"""
query = "O que é system prune?"
docs_score = db.similarity_search_with_score(query=query, distance_metric="cos", k = 6)

print(docs_score)
print("")
print(docs_score[0])










