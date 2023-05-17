"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro
"""

# Isto é quando usas o arquivo .env:
import openai 
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY']  
Eddy_API_KEY_Cohere = os.environ["COHERE_API_KEY"]
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]
Eddy_API_KEY_SerpApi = os.environ["SERPAPI_API_KEY"]
Eddy_API_KEY_WolframAlpha = os.environ["WOLFRAM_ALPHA_APPID"]


"""
Criando um Agente
=================
"""
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI


llm = OpenAI(temperature=0)


"""
Loading Tools
=============
"""
tools = load_tools(["serpapi", "llm-math"],
                   llm=llm
                  )


# O que é uma ferramenta como esta:
print(tools[1].name, tools[1].description)

print(tools[1])





