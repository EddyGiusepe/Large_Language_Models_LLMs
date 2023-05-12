"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro


Aprimore os recursos de pesquisa na web do seu chatbot com Langchain e SerpAPI
==============================================================================

Como o título já diz, aqui vamos aprender como o Langchain aprimora a funcionalidade do ChatGPT, 
integrando recursos de pesquisa na web perfeitos para projetos de IA.

Exploraremos como aprimorar nossos ChatGPTs com recursos de pesquisa na web usando Langchain e SerpAPI.


Instala as dependências
-----------------------

* LangChain:  Uma ferramenta poderosa que fornece uma cadeia AgentExecutor, permitindo que seu 
              modelo ChatGPT realize uma pesquisa na web e retorne informações relevantes.

* OpenAI: O cliente oficial da API OpenAI, permitindo que você interaja com o GPT-3 ou GPT-4.


LangChain e SerpAPI
-------------------
Langchain é uma biblioteca Python que permite construir uma cadeia AgentExecutor para seu modelo 
ChatGPT. Ele permite que seu chatbot realize ações específicas, como realizar pesquisas na web, 
analisar os resultados e gerar respostas apropriadas com base nas informações coletadas.


SerpAPI é uma API de Search Engine Results Page (SERP) que fornece uma maneira fácil de recuperar 
os resultados do mecanismo de pesquisa em um formato estruturado. Ao integrar SerpAPI com Langchain, 
você pode habilitar seu modelo ChatGPT para pesquisar na web e obter informações relevantes sem 
analisar manualmente as páginas da web.

"""
# Isto é quando usas o arquivo .env: 
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY']  
Eddy_API_KEY_SerpApi = os.environ["SERPAPI_API_KEY"]
Eddy_API_KEY_WolframAlpha = os.environ["WOLFRAM_ALPHA_APPID"]

from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI


llm = OpenAI(temperature=0) # Resultados determinísticos = 0

tools = load_tools(["serpapi"],
                   llm=llm
                  )

agent = initialize_agent(tools,
                         llm,
                         agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

#response = agent.run("De acordo com o MarketWatch, qual é o preço do S&P 500 Future Dec 2024?")
response = agent.run("Quem é o presidente do Perú?")
print(response)
