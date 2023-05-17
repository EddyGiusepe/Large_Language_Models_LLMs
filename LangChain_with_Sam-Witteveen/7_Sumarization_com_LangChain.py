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

docs = [Document(page_content=t) for t in texts]
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

"""
Você vai reparar que temos dois Prompt. Um que resume cada parte e o outro Prompt que combina esses resumos.
"""
# Um Prompt que resume cada parte:
print(chain.llm_chain.prompt.template)

# Outro Prompt que combina as partes:
chain.combine_document_chain.llm_chain.prompt.template

"""
Sendo um pouco mais detalhista:
"""
chain = load_summarize_chain(llm, 
                             chain_type="map_reduce",
                             verbose=True
                             )

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, 
                             width=100,
                             break_long_words=False,
                             replace_whitespace=False)
print(wrapped_text)


"""
Resumindo com a Chain 'stuff'

Stuff:
------
Stuffing é o método mais simples, no qual você simplesmente coloca todos os dados relacionados
no prompt como contexto para passar para o modelo de linguagem. Isso é implementado no LangChain
como StuffDocumentsChain.

Prós:
===== Faz apenas uma única chamada para o LLM. Ao gerar texto, o LLM tem acesso a todos os dados de uma só vez.

Contras:
======== A maioria dos LLMs possui um comprimento de contexto e, para documentos grandes
         (ou muitos documentos), isso não funcionará, pois resultará em um prompt maior
         que o comprimento do contexto.

A principal desvantagem desse método é que ele funciona apenas com pedaços menores de dados. Depois de
trabalhar com muitos dados, essa abordagem não é mais viável. As próximas duas abordagens são projetadas
para ajudar a lidar com isso.

Link parte 1 --> https://colab.research.google.com/drive/1FJ7-nhTktyMSsbxI6CHye1tsfrY1Khqi?usp=sharing#scrollTo=OzCHODNOPKnO

Link parte 2 --> https://colab.research.google.com/drive/1uS_ARUtnrWH1PUAdo1yYe0O48-0YZE3s?usp=sharing
"""














