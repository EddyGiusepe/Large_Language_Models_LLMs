"""
Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro

Objetivo: Neste script estudamos o uso da "Function calling" da OpenAI,
          a qual fornece a saída em formato JSON. Pegamos o exemplo que 
          eles fornecem.    

Modelo de execução: 
                   $ python 1_Function_calling.py
"""
import openai
import json
import os
from dotenv import find_dotenv, load_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.getenv('OPENAI_API_KEY')


# Exemplo de função fictícia codificada para retornar o mesmo clima
# Em produção, pode ser sua API de back-end ou uma API externa:
def get_current_weather(location, unit="fahrenheit"):
    """Obter o clima atual em um determinado local"""
    weather_info = {
        "location": location,
        "temperature": "12",
        "unit": unit,
        "forecast": ["ensolarado", "ventoso"],
    }
    return json.dumps(weather_info)


def run_conversation():
    # Step 1: envie a conversa e as funções disponíveis para GPT
    messages = [{"role": "user", "content": "Como é o clima em Lima?"}]
    functions = [{"name": "get_current_weather",
                  "description": "Obter o clima atual em um determinado local",
                  "parameters": {"type": "object",
                                 "properties": {"location": {"type": "string","description": "O departamento e cidade, por exemplo. Cercado de Lima, Lima",},
                                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},},
                                 "required": ["location"],
                                },
                 }
                ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto é o padrão, mas seremos explícitos
    )
    
    response_message = response["choices"][0]["message"]

    # Step 2: verifique se o GPT queria chamar uma função
    if response_message.get("function_call"):
        # Step 3: chame a função
        # Note: A resposta JSON pode nem sempre ser válida; certifique-se de lidar com erros:
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # Apenas uma função neste exemplo, mas você pode ter várias
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = fuction_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )

        # Step 4: Envie as informações sobre a chamada de função e a resposta da função para GPT
        messages.append(response_message)  # Estender a conversa com a resposta do assistente
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # Estender conversa com resposta de função:
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # Obtenha uma nova resposta do GPT, onde pode ver a resposta da função
        return second_response


print(run_conversation())
