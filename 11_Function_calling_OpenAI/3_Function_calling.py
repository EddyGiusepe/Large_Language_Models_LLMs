"""
Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro

Objetivo: Neste script estudamos o uso da "Function calling" da OpenAI,
          a qual fornece a saída em formato JSON. Pegamos o exemplo que 
          eles fornecem.    

Modelo de execução: 
                   $ python 3_Function_calling.py
"""
import openai
import json
import os
from dotenv import find_dotenv, load_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.getenv('OPENAI_API_KEY')

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[{"role": "user", "content": "Classifique a frase a seguir. Frase:\n Super chateado, não consigo dormir agora!"}],
    functions=[{"name": "classify_user_input",
                "description": "Classifique as informações relevantes da frase.",
                "parameters": {"type": "object",
                               "properties": {"tone": {"type": "string",
                                                       "description": "O tom da frase."
                                                      },
                                              "sentiment": {"type": "string",
                                                            "description": "O sentimento da frase."
                                                           },
                                              "language": {"type": "string",
                                                           "description": "O idioma da frase."
                                                          }
                                             },
                                "required": ["tone", "sentiment", "language"]
                              }
               }
              ],       
    function_call={"name": "classify_user_input"}         
                                        )


output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])
print(output)
