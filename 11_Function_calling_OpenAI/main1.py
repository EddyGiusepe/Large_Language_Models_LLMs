"""
Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro

Objetivo: Neste script usamos a "Function calling" da OpenAI,
          a qual fornece a saída em formato JSON.    

Modelo de execução: 
                   $ python main1.py
"""
import openai
import json
import os
from dotenv import find_dotenv, load_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.getenv('OPENAI_API_KEY')

def Function_calling(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": f"Na frase faz um Reconhecimento de Entidades Nomeadas (NER): {query}"}],
        functions=[{"name": "get_entities_in_user_query",
                    "description": "Classifique as entidades relevantes da frase.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ORG": {"type": "string",
                                    "description": "Entidade nomeada do tipo ORGANIZAÇÃO (ORG)."
                                    },
                            "PER": {"type": "string",
                                        "description": "Entidade nomeada do tipo PESSOA (PER)"
                                },
                            "LOC": {"type": "string",
                                        "description": "Entidade nomeada do tipo LOCALIZAÇÃO (LOC)"
                                }
                                    },
                    "required": ["ORG", "PER", "LOC"]
                                }
                    }
                ],
        function_call={"name": "get_entities_in_user_query"}         
    )


    output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])
    return output

print(Function_calling("A OpenAI tem sede em São Francisco e o lider é o engenheiro Jhonatan!"))
