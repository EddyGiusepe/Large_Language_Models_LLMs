"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro
"""
####################################
# Brincando a imprimir o calendário:
import calendar

year = 2023
month = 4
x = calendar.month(year, month)
print(x)
####################################

# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]

"""
Chain com PAL Mat
-----------------
"""
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import PALChain

# Istanciamos nosso Modelo:
llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens=512)

# A nossa Chain:
pal_chain = PALChain.from_math_prompt(llm,
                                      verbose=True
                                     )

# Temos dois exemplos para calcular os resultados:
question_1 = """Jan tem três vezes mais animais de estimação do que Márcia. Márcia tem dois
                animais de estimação a mais que Cindy. Se Cindy tem quatro animais de estimação,
                quantos animais de estimação os três têm?
             """

question_2 = """O refeitório tinha 23 maçãs. Se eles usaram 20 no almoço e compraram mais 6,
                quantas maçãs eles têm?
             """

user_input = input("Digite o seu problema de Matemática? ") 
print(pal_chain.run(user_input))


"""
API Chains - OpenMeteo - Informação meteorológica
-------------------------------------------------

nota --> Pode gerar erros com base no comprimento do retorno da API
"""
from langchain import OpenAI
from langchain.chains.api.prompt import API_RESPONSE_PROMPT
from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate

llm = OpenAI(temperature=0,
             max_tokens=100)


from langchain.chains.api import open_meteo_docs

chain_new = APIChain.from_llm_and_api_docs(llm,
                                           open_meteo_docs.OPEN_METEO_DOCS,
                                           verbose=True
                                          )

user_input = input("Digite o sua pergunta METEOROLÓGICA: ") # Qual é a Temperatura na cidade de Vitória-ES?
print(chain_new.run(user_input))
