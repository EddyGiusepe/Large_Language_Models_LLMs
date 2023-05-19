"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro

Comparando e Avaliando LLMs
===========================

Neste script vamos a comparar vários modelos observando suas saídas para uma variedade de tarefas. Esta é uma área que o 
LangChain também está adicionando. Os modelos cobertos incluem GPT-3, ChatGPT --> 'gpt-3.5-turbo', Flan-20B, Flan-T5-XL, 
Cohere-command-xl.

$ !pip -q install langchain huggingface_hub openai==0.27.2 google-search-results tiktoken cohere

$ !pip show langchain
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


# Definimos a variável de ambiente CUDA_VISIBLE_DEVICES como um valor inválido para a GPU ou seja CPU:
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0" para usar GPU "-1" para CPU


"""
Configurando Modelos Flan
=========================
"""
from langchain import PromptTemplate, HuggingFaceHub, LLMChain


flan_20B = HuggingFaceHub(repo_id="google/flan-ul2", 
                         model_kwargs={"temperature":0.1, 
                                       "max_new_tokens":100}
                         ) 


flan_t5xxl = HuggingFaceHub(repo_id="google/flan-t5-xxl", 
                         model_kwargs={"temperature":0.1, 
                                       "max_new_tokens":100}
                         ) 

# unfortunately not working
# GPTNeoXT_20B = HuggingFaceHub(repo_id="togethercomputer/GPT-NeoXT-Chat-Base-20B", 
#                          model_kwargs={"temperature":0.0, 
#                                        "max_new_tokens":200}
#                          )


# unfortunately not working
# bloom7B = HuggingFaceHub(repo_id="bigscience/bloom-7b1", 
#                          model_kwargs={"temperature":0.0, 
#                                        "max_new_tokens":200}
#                          )


# gpt_j6B = HuggingFaceHub(repo_id="EleutherAI/gpt-j-6B", 
#                          model_kwargs={"temperature":0.1, 
#                                        "max_new_tokens":100},
#                          verbose=False                
#                         )


"""
Configurando Modelos OpenAI
===========================
"""
from langchain.llms import OpenAI, OpenAIChat

chatGPT_turbo = OpenAIChat(model_name='gpt-3.5-turbo',
                           temperature=0.0, 
                           max_tokens = 150,
                           verbose=False
                          )

gpt3_davinici_003 = OpenAI(model_name='text-davinci-003',
                           temperature=0.0, 
                           max_tokens = 150,
                           verbose=False
                          )


"""
Configurando Modelos Cohere
===========================
"""
from langchain.llms import Cohere

cohere_command_xl = Cohere(model='command-xlarge',
                           temperature=0.1, 
                           max_tokens = 150
                          )



cohere_command_xl_nightly = Cohere(model='command-xlarge-nightly',
                                   temperature=0.1, 
                                   max_tokens = 150
                                  )



"""
Configurando um Laboratório de comparação
=========================================
"""
from langchain.model_laboratory import ModelLaboratory
from langchain.prompts import PromptTemplate

template = """Pergunta: {question}

Resposta: Vamos pensar passo a passo."""

prompt = PromptTemplate(template=template,
                        input_variables=["question"])



# Nosso Laboratório:
lab = ModelLaboratory.from_llms([chatGPT_turbo, 
                                 gpt3_davinici_003,
                                 #gpt_j6B, 
                                 flan_20B,
                                 flan_t5xxl, 
                                 cohere_command_xl, 
                                 cohere_command_xl_nightly
                                 ],
                                 prompt=prompt
                                )


# Vamos a comparar!
print(lab.compare("Qual é o oposto de para cima?"))


print(lab.compare("Responda a seguinte pergunta raciocinando passo a passo. A lanchonete tinha 23 maças. \
Se eles usaram 20 no almoço e compraram mais 6, quantas maçãs eles têm?"))



"""Mudando o PROMPT"""
template = """Você é um contador de histórias criativo que pode escrever contos maravilhosos e interessantes: {question}

História:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

lab = ModelLaboratory.from_llms([
                                 chatGPT_turbo, 
                                 gpt3_davinici_003,
                                 #gpt_j6B, 
                                 flan_20B,
                                 flan_t5xxl, 
                                 cohere_command_xl, 
                                 cohere_command_xl_nightly
                                 ],
                                 prompt=prompt
                                )


lab.compare('''Escreva uma história triste sobre uma cenoura chamada Jason. A história deveria \
começar com a cenoura sendo uma espécie de atleta profissional e terminar com a \
cenoura tendo seu coração partido.''')




"""Mudando o PROMPT"""
template = """Responda à pergunta com o melhor de suas habilidades, mas se não tiver certeza, responda que não sabe: {question}

Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

lab = ModelLaboratory.from_llms([
                                 chatGPT_turbo, 
                                 gpt3_davinici_003,
                                 #gpt_j6B, 
                                 flan_20B,
                                 flan_t5xxl, 
                                 cohere_command_xl, 
                                 cohere_command_xl_nightly
                                 ],
                                 prompt=prompt
                                )


lab.compare('''Eu estou andando de bicicleta. Os pedais estão se movendo rapidamente. Olho no espelho e não me mexo. Por que é isso?''')


"""
Extração de fatos
=================
"""
template = """{question}

Resposta:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

lab = ModelLaboratory.from_llms([
                                 chatGPT_turbo, 
                                 gpt3_davinici_003,
                                 #gpt_j6B, 
                                 flan_20B,
                                 flan_t5xxl, 
                                 cohere_command_xl, 
                                 cohere_command_xl_nightly
                                 ],
                                 prompt=prompt
                                )



lab.compare('''Please answer the question:\n
Who is the OnePlus COO?\n\n
Output in the format: [first_name, surname]\n\n

Smartphone makers searched for a way forward at MWC 2023
Foldables, 6G, light shows -- there are a lot of ideas floating around, but no one has cracked the code
The slowdown was inevitable, of course. Nothing stays hot forever — especially in this industry. By tech standards, smartphones have had a good run, but the last few years have seen device makers searching for the magic bullet to help the sales slide reverse course. The arrival of 5G was a nice reprieve, but next-generation telecom standards don’t arrive every year.

“I personally think foldables are supply chain-driven innovation and not consumer insights,” Pei said. “Somebody invents OLED, and they can make a lot of money, because it’s a great technology. Then after a few years, a lot more companies make that, so they need to lower their prices. So they need to figure out what else they can sell at a higher margin. They develop flexible OLEDs, which they can sell at a higher price.”
It’s hard not to be cynical about this stuff sometimes. Ditto for concept devices, though as I noted in my “ode to weird tech” post, as someone who follows this stuff for a living, I’m a fan of weirdness for weirdness sake, be it the rollable Motorola Rizr screen or the OnePlus glowing cooling fluid. Certainly following the automotive industry’s lead of creating concept devices is a trend that is likely to only become more pervasive.

OnePlus COO Kinder Liu told me this week that gauging consumer interest is one of the “multiple reasons” his company is engaging with the concept. He added, “Also, we want to encourage continuous innovation inside our company.”

Pretty much everyone I engaged with this week echoed the sentiment that smartphones are in a rut. For the first time, however, it’s not a foregone conclusion that there’s a way of getting out.
''')



lab.compare('''Please answer the question:\n
What is a supply chain driven innovation?\n\n

Smartphone makers searched for a way forward at MWC 2023
Foldables, 6G, light shows -- there are a lot of ideas floating around, but no one has cracked the code
The slowdown was inevitable, of course. Nothing stays hot forever — especially in this industry. By tech standards, smartphones have had a good run, but the last few years have seen device makers searching for the magic bullet to help the sales slide reverse course. The arrival of 5G was a nice reprieve, but next-generation telecom standards don’t arrive every year.

“I personally think foldables are supply chain-driven innovation and not consumer insights,” Pei said. “Somebody invents OLED, and they can make a lot of money, because it’s a great technology. Then after a few years, a lot more companies make that, so they need to lower their prices. So they need to figure out what else they can sell at a higher margin. They develop flexible OLEDs, which they can sell at a higher price.”
It’s hard not to be cynical about this stuff sometimes. Ditto for concept devices, though as I noted in my “ode to weird tech” post, as someone who follows this stuff for a living, I’m a fan of weirdness for weirdness sake, be it the rollable Motorola Rizr screen or the OnePlus glowing cooling fluid. Certainly following the automotive industry’s lead of creating concept devices is a trend that is likely to only become more pervasive.

OnePlus COO Kinder Liu told me this week that gauging consumer interest is one of the “multiple reasons” his company is engaging with the concept. He added, “Also, we want to encourage continuous innovation inside our company.”

Pretty much everyone I engaged with this week echoed the sentiment that smartphones are in a rut. For the first time, however, it’s not a foregone conclusion that there’s a way of getting out.
''')
