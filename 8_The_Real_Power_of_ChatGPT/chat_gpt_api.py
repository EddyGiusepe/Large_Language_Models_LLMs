# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
print('Carregando a minha chave Key: ', load_dotenv())
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 

import openai
openai.api_key = Eddy_API_KEY_OpenAI

selected_model = "gpt-3.5-turbo"

def basic_generation(user_prompt):
    completion = openai.ChatCompletion.create(model=selected_model,
                                              messages=[{"role": "user", "content": user_prompt}]
                                             )
    
    response = completion.choices[0].message.content
    return response
