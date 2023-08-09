"""
Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro

An Introduction to OpenAI Function Calling
==========================================
Este script está baseado na publicação de "David Hundley".

link de estudo --> https://towardsdatascience.com/an-introduction-to-openai-function-calling-e47e7cd7680e
"""
import openai
import json
import os
from dotenv import find_dotenv, load_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.getenv('OPENAI_API_KEY')

# Carregando o texto "Sobre mim" do arquivo local:
with open('./perfil_Eddy.txt', 'r') as f:
    sobre_mim = f.read()

print(sobre_mim)

"""Usando a API da OpenAI com um PROMPT
   ====================================
"""

# Criando um prompt para extrair o máximo de informações de "Sobre mim" como um objeto JSON:
sobre_mim_prompt = f""" 
Por favor, extraia as informações como um objeto JSON. Por favor, procure as seguintes informações. 
Nome 
Cargo 
Empresa 
Número de filhos como um único inteiro 
Hobby
Onde estudou
Casado 

Este é o corpo do texto para extrair as informações de: 
```{sobre_mim}```
""" 

# Usamo a API da OpenAI (ChatGPT - gpt-3.5-turbo)
openai_response = openai.ChatCompletion.create(model = 'gpt-3.5-turbo',
                                               messages = [{'role': 'user', 'content': sobre_mim_prompt}]
                                              )

# Carregando a resposta com um objeto JSON:
json_response = json.loads(openai_response['choices'][0]['message']['content'])
#print(json_response)


"""🤗 Usando a API da OpenAI com Function Calling 🤗
   =================================================
"""
# Definimos a nossa função para extrair informações pessoais:
def extract_person_info(nome: str, cargo: str, casado: bool):
    '''
    Imprime informações básicas "Sobre mim"

    Inputs:
        nome (str): Nome da pessoa
        cargo (str): profissão da pessoa
        casado (bool): Se a pessoa é casada.
    '''
    
    print(f'Esta pessoa se chama {nome}. O cargo dele é {cargo}, e ele é {casado}.')


# Definindo como queremos que o ChatGPT chame nossas funções personalizadas
my_custom_functions = [
    {
        'name': 'extract_person_info',
        'description': 'Obtenha informações "Sobre mim" do corpo do texto de entrada',
        'parameters': {
            'type': 'object',
            'properties': {
                'nome': {
                    'type': 'string',
                    'description': 'Nome da pessoa'
                },
                'cargo': {
                    'type': 'string',
                    'description': 'Profissão da pessoa'
                },
                'casado': {
                    'type': 'boolean',
                    'description': 'Se a pessoa é casada'
                }
            }
        }
    }
]


# Usando novamente a API da OpenAI:
openai_response = openai.ChatCompletion.create(model = 'gpt-3.5-turbo',
                                               messages = [{'role': 'user', 'content': sobre_mim}],
                                               functions = my_custom_functions,
                                               function_call = 'auto'
)

#print(openai_response)
print("🤗🤗🤗")
print(openai_response['choices'][0]["message"]["function_call"]["arguments"])


"""
Então, o que acontece quando enviamos um prompt que não corresponde a nenhuma de nossas funções personalizadas? 
Simplificando, o padrão é o comportamento típico, como se a chamada de função não existisse. 
Vamos testar isso com um prompt arbitrário: "Qual é a altura da Torre Eiffel?" 🧐
"""

# Getting the response back from ChatGPT (gpt-3.5-turbo)
openai_response = openai.ChatCompletion.create(
    model = 'gpt-3.5-turbo',
    messages = [{'role': 'user', 'content': 'Qual é a altura da Torre Eiffel?'}],
    functions = my_custom_functions,
    function_call = 'auto'
)

#print(openai_response)
json_response = openai_response['choices'][0]['message']['content']
print(json_response)
