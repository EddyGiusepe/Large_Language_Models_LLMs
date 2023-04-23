"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro


Resumo com LangChain
--------------------
Histórico
Desafios
Ajuste fino (Fine-tuning)
Ajuste de instruções
"""

# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]


"""
Configurando a cadeia de resumo
-------------------------------
"""
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0,
             verbose=True
            )

text_splitter = CharacterTextSplitter()

# Carregando nosso Documento:
with open('./Newtonian_Physics.txt') as f:
    how_to_win_friends = f.read()
texts = text_splitter.split_text(how_to_win_friends)


print(len(texts))


from langchain.docstore.document import Document

docs = [Document(page_content=t) for t in texts[:4]]
#print(docs)

"""
3 tipos de cadeias "CombineDocuments"
-------------------------------------
Resumo simples com map_reduce

Map Reduce :
----------

Este método envolve um PROMPT inicial em cada bloco de dados (para tarefas de resumo, pode
ser um resumo desse bloco; para tarefas de resposta a perguntas, pode ser uma resposta
baseada apenas nesse bloco). Em seguida, um PROMPT diferente é executado para combinar todas
as saídas iniciais. Isso é implementado no LangChain como o 'MapReduceDocumentsChain'.

Prós:
===== Pode escalar para documentos maiores (e mais documentos) do que 'StuffDocumentsChain'.
      As chamadas para o LLM em documentos individuais são independentes e, portanto, podem ser paralelizadas.

Contras:
======== Requer muito mais chamadas para o LLM do que 'StuffDocumentsChain'. Perde algumas
         informações durante a chamada de combinação final.
"""
from langchain.chains.summarize import load_summarize_chain
import textwrap

chain = load_summarize_chain(llm, 
                             chain_type="map_reduce")


output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print(wrapped_text)

# Para resumir cada parte:
print(chain.llm_chain.prompt.template)


