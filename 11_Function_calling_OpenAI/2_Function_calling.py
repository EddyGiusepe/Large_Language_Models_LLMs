"""
Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro

Objetivo: Neste script estudamos o uso da "Function calling" da OpenAI,
          a qual fornece a saída em formato JSON. Pegamos o exemplo que 
          eles fornecem.    

Modelo de execução: 
                   $ python 2_Function_calling.py

Link de estudo: https://medium.com/dev-bits/a-clear-guide-to-openai-function-calling-with-python-dcbc200c5d70
"""
import openai
import json
import os
from dotenv import find_dotenv, load_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.getenv('OPENAI_API_KEY')


from typing import List
from pydantic import BaseModel

class StepByStepAIResponse(BaseModel):
    title: str
    steps: List[str]

schema = StepByStepAIResponse.schema() # retorna um dict como esquema JSON

# O conteúdo do schema é parecido com:

"""
{
    'title': 'StepByStepAIResponse',
    'type': 'object',
    'properties': {'title': {'title': 'Title', 'type': 'string'},
    'steps': {'title': 'Steps', 'type': 'array', 'items': {'type': 'string'}}},
    'required': ['title', 'steps']
}
"""
print(schema)

response = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613",
                                        messages=[{"role": "user", "content": "Como posso preparar o ceviche peruano?"} # "Explain how to assemble a PC"
                                                 ],
                                        functions=[{"name": "get_answer_for_user_query",
                                                    "description": "Obtenha a resposta do usuário em uma série de passos (steps).",
                                                    "parameters": StepByStepAIResponse.schema()
                                                   }
                                                  ],
                                        function_call={"name": "get_answer_for_user_query"}
                                       )

output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])
print(output)
# output content
"""
{
    'title': 'Steps to assemble a PC',
    'steps': [
        '1. Gather all necessary components',
        '2. Prepare the PC case',
        '3. Install the power supply',
        '4. Mount the motherboard',
        '5. Install the CPU and CPU cooler',
        '6. Install RAM modules',
        '7. Install storage devices',
        '8. Install the graphics card',
        '9. Connect all cables',
        '10. Test the PC'
    ]
}
"""

sbs = StepByStepAIResponse(**output)
#Now access sbs.title, sbs.steps in your code
print("")
print(sbs.title)
print("")
print(sbs.steps)
