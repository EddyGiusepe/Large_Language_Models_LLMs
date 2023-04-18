"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro
"""

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain


# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
print('Carregando a minha chave Key: ', load_dotenv())
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 


llm = OpenAI(temperature=0.0)


template = """
           Quero que você atue como consultor de nomes para novas empresas.
           Aqui estão alguns exemplos de bons nomes de empresas:

         * search engine, Google
         * social media, Facebook
         * video sharing, YouTube
           
           O nome deve ser curto, cativante e fácil de lembrar.

        Qual é um bom nome para uma empresa que faz {product}?   
"""
prompt = PromptTemplate(
    input_variables=["product"],
    template= template #"Qual é um bom nome para uma empresa que faz {product}?",
                       )

# print(prompt.format(product="Meias coloridas"))
# text = prompt.format(product="Meias coloridas")
# print(llm(text))


# Crio uma Chain:
chain = LLMChain(llm=llm,
                 prompt=prompt
                )

user_input = input("Digite qual é a sua ideia de Produto? ") 
print(chain.run(user_input))
