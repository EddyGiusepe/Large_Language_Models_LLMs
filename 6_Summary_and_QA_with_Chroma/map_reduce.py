"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro
"""
"""
map_reduce
----------
Baseado no tutorial de echohive:
https://www.youtube.com/watch?v=gY25ddXWNw4
"""
from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

# Deve instalar --> pip install chromadb
# Deve instalar --> pip install tiktoken
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain import OpenAI

# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]


# Diretório persistente para o chromadb, de modo que o armazenamento de vetores seja armazenado localmente.
persist_directory = 'db'
embeddings = OpenAIEmbeddings()

# Verificamos se o DB já existe:
if not os.path.exists(persist_directory):
    # Carregar Newtonian_Physics.txt 
    with open("./Newtonian_Physics.txt") as f:
        book = f.read()       

    print("Carregando o .txt")
    loader = TextLoader("./Newtonian_Physics.txt")

    documents = loader.load()
    #print(documents[0].page_content[:])

    text_splitter = CharacterTextSplitter(chunk_size=60,
                                          chunk_overlap=0
                                         )
    docs = text_splitter.split_documents(documents)
    #print(docs[0].page_content[:])

    print("embeddings book_ascii.txt")
    db = Chroma.from_documents(docs,
                                embeddings,
                                persist_directory=persist_directory
                                )
else:    
    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings
               )

llm_model = OpenAI(temperature=0,
                   verbose=True,
                   max_tokens=150
                  ) 

# Lembrando que o map_reduce, requer da instalação de tiktoken.
chain = load_qa_chain(llm=llm_model,
                      chain_type="map_reduce",
                      return_map_steps=True
                     )

while True:
    # Para rastrear os Tokens:
    with get_openai_callback() as cb:
        query = input("Digite a sua query: ")
        docs = db.similarity_search(query)
        result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        print(result['output_text'])
        #print(result['intermediate_steps])
        #print(docs)
        print("Tokens usados: ", cb.total_tokens)

# Você pode customizar ainda o prompt para map_redeuce, ver documentação.