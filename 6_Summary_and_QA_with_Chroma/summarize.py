"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro
"""
"""
Langchain Summary e QA com Chromadb usando OpenAI Embeddings e GPT 3 com contagem de token
------------------------------------------------------------------------------------------
Baseado no tutorial de echohive
"""
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback

# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# Nosso modelo:
llm = OpenAI(temperature=0.0,
             verbose=True,
             max_tokens=150
            )

text_splitter =  CharacterTextSplitter()

with open("./Newtonian_Physics.txt") as f:
    space = f.read()


texts = text_splitter.split_text(space)

print("Comprimento da lista de textos: ", len(texts))

docs = [Document(page_content=t) for t in texts] # pode limitar: texts[:3]

# Criamos a nossa Chain:
chain = load_summarize_chain(llm, chain_type="map_reduce") # map_reduce --> reduz cada chunk primeiro 

with get_openai_callback() as cb:
    result = chain.run(docs)
    print(result)
    print("Tokens usados: ", cb.total_tokens)

