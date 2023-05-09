"""
Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro

A ideia: 
        Aqui vamos conversar com nossos documentos de texto. Usamos OpenAI e LangChain
        para transformar documentos estáticos em conversas interativas e seguidamente
        concatenar com outra cadeia que fará o Few-Shot Learning.
Objetivo: 
         Simular um fluxo, end-to-end, onde tenhamos um Dataset armazenado e un dataset pequeno
         próprio de uma conversa entre o Usuário e o Assistente.    
"""

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]


with open("/home/eddygiusepe/1_Eddy_Giusepe/3_estudando_LLMs/Large_Language_Models_LLMs/4_Chat_with_Your_Documents-OpenAI_with_LangChain/RESUMINHO_estado_da_Uniao.txt") as f:
    state_of_the_union = f.read()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

embeddings = OpenAIEmbeddings()

docsearch = Chroma.from_texts(texts,
                              embeddings,
                              metadatas=[{"source": f"Text chunk {i} of {len(texts)}"} for i in range(len(texts))],
                              persist_directory="db"
                            )

docsearch.persist()

docsearch = None
