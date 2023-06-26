"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Iisdro

Objetivo: Este script realiza o resumo de seu documento texto 🤗.

Executar este script:
                     $ python main.py
"""
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain # Para resumo
from langchain.text_splitter import CharacterTextSplitter # Para divisão de texto
from langchain.chains.mapreduce import MapReduceChain # Para resumir o documento
from langchain.docstore.document import Document
import textwrap # Formata o resultado

import os
import openai
from dotenv import load_dotenv
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 


llm = OpenAI(model_name="text-davinci-003")

text_splitter = CharacterTextSplitter()

with open("Newtonian_Physics.txt") as f:
    data = f.read()

texts = text_splitter.split_text(data)

docs = [Document(page_content=t) for t in texts[:3]]

chain = load_summarize_chain(llm, chain_type="map_reduce")
output_summary = chain.run(docs)

wrapped_text = textwrap.fill(output_summary, width=120)
print(wrapped_text)
