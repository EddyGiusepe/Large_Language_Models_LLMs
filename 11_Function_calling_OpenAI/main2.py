"""
Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro

Objetivo:
========= Com este script extraímos ENTIDADES (NER). Para isso
          usamos a API da OpenAI e Function calling. A saída é
          um formato JSON.  

Método de xecução:
================== Você pode executar no Terminal, assim:

                  $ uvicorn main2:app --host 0.0.0.0 --port 8000 --reload
                  ou
                  $ python main2.py
"""
import openai
import json
import os
from fastapi import FastAPI
from pydantic import BaseModel

from dotenv import find_dotenv, load_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.getenv('OPENAI_API_KEY')

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    entities: dict

app = FastAPI(title='🤗 NER com a API da OpenAI e Function calling 🤗',
              version='1.0.0',
              description="""Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro""")

@app.post("/process_query")
async def Function_calling(query: QueryRequest):
    prompt = f"""Você deve atuar como um especialista em ciência de dados e Processamento de linguagem \
                 Natural (NLP). A seguir realiza o Reconhecimento de Entidades Nomeadas (NER) do tipo: \
                 ORGANIZAÇÃO, PESSOA e LOCALIZAÇÃO em: {query}"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": prompt}],
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
        temperature=0.0, # Parâmetro para diminuir a natureza estocástica. 
        function_call={"name": "get_entities_in_user_query"}         
    )


    output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])
    return QueryResponse(entities=output)


# Executar a API usando o servidor Uvicorn:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
