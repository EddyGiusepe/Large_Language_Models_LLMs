"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro


LangChain Agents = Tools + Chains
=================================

Neste script, examinamos os agentes LangChain e como eles permitem que você use várias ferramentas 
e cadeias em um aplicativo LLM, permitindo que seu LLM decida sobre a próxima ferramenta de entrada 
a ser usada com base na entrada do usuário.

Você deve instalar os seguintes pacotes:

$ !pip -q install langchain huggingface_hub openai google-search-results tiktoken wikipedia
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
#print(tools[1].name, tools[1].description)

#print(tools[1])

"""
Inicializando o agente
======================
"""
agent = initialize_agent(tools, 
                         llm, 
                         agent="zero-shot-react-description", 
                         verbose=False)

agent.agent.llm_chain.prompt.template


# Consulta LLM padrão
#print(agent.run("Olá, como você está hoje?"))


print(agent.run("Quem é o presidente dos Estados Unidos?"))


"""
Novo agente
===========
"""
tools = load_tools(["serpapi", "llm-math", "wikipedia", "terminal"],
                   llm=llm
                  )


agent = initialize_agent(tools, 
                         llm, 
                         agent="zero-shot-react-description", 
                         verbose=False
                        )


agent.agent.llm_chain.prompt.template


# print(agent.run("Quem é o chefe da DeepMind?"))


# print(agent.run("O que é DeepMind?"))


# print(agent.run("Se eu elevar ao quadrado o número do endereço da DeepMind, que resposta obtenho?"))


# print(agent.run("Quais arquivos estão no meu diretório atual?"))

print(agent.run("Meu diretório atual tem um arquivo sobre a Califórnia?"))