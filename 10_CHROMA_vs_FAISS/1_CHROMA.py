"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro

CHROMA
======

Chroma é um banco de dados para criar aplicativos de IA com Embeddings.

Aqui vou estudar e analisar certos parâmetros relevantes na hora de realizar a pesqusia 
por similaridade em texto.
"""
# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]


from langchain.chains import RetrievalQA # Tem que atualizar --> pip install langchai==0.0.137
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# As seguinte biblioteca é para usar os Embeddings do HuggingFace:
from langchain.embeddings import HuggingFaceEmbeddings


# Definimos a variável de ambiente CUDA_VISIBLE_DEVICES como um valor inválido para a GPU ou seja CPU:
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # "0" para usar GPU "-1" para CPU


# Carregando meu documento em PDF:
#loader = PyPDFLoader("./Eddy_Running_the_Vale_API.pdf")
#documents = loader.load()


loader = TextLoader("./macdonalds_euamopontos.txt")
documents = loader.load()


# Dividir os documentos em chunks:
#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
texts = text_splitter.split_documents(documents=documents) # Para .pdf e .txt 


persist_directory = './chromadb'

# Selecione que EMBEDDINGS quer usar:
#embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings()

# Crie o vectorestore para usar como índice (index):
db = Chroma(collection_name='eddy_1',
            embedding_function=embeddings,
            persist_directory=persist_directory)


db.add_documents(documents=texts, embeddings=embeddings)
db.persist()

""""
Lembrar que o SCORE na hora de realizar a Pesquisa por Similaridade é um float que está
relacionado com a Distância. Default é DISTÂNCIA EUCLIDIANA 🧐
"""
# #query = """A McDonald’s tem algum colaborador infetado?"""
# query = """Como são criados os frangos?"""
# docs_score = db.similarity_search_with_score(query=query, distance_metric="cos", k = 4)
# #docs_score = db.similarity_search(query=query)

# print(docs_score[0][0].page_content)
# print("")
# print(docs_score)
# print("Os scores de distância: ")
# print(docs_score[0][1])
# print(docs_score[1][1])
# print(docs_score[2][1])
# print(docs_score[3][1])

print("Digite a sua pergunta para similarity search: ")
while True:
    query = input("Pergunta do usuário: ")
    docs_score = db.similarity_search_with_score(query=query, distance_metric="cos", k=4)
    print(docs_score)
    print("")
    resposta = docs_score[0][0].page_content
    print("\033[033mA resposta mais SIMILAR é: \033[m", resposta)

    if not query:
        break


