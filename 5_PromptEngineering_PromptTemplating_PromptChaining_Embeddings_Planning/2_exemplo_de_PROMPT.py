# Isto é quando usas o arquivo .env:
import openai 
from dotenv import load_dotenv
import os
import spacy

print('Carregando a Chave: ', load_dotenv())
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY']  
Eddy_API_KEY_Cohere = os.environ["COHERE_API_KEY"]
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]
Eddy_API_KEY_SerpApi = os.environ["SERPAPI_API_KEY"]
Eddy_API_KEY_WolframAlpha = os.environ["WOLFRAM_ALPHA_APPID"]

# Adicione isto para usar a sua have:
openai.api_key = Eddy_API_KEY_OpenAI

# Carregue o modelo da linguagem do spaCy para português
nlp = spacy.load("pt_core_news_md")

# Função para extrair entidades nomeadas de uma frase
def extrair_entidades(frase):
    doc = nlp(frase)
    return [(ent.text, ent.label_) for ent in doc.ents]


prompt = f"""
Esta é uma conversa com a AI. Você deve extrair as entidades com \
a função definida anteriormente (com o spacy), para seguidamente gerar uma \
resposta com contexto para o usuário.  
"""

def get_completion(prompt, model ="gpt-3.5-turbo", temperature =0):

    messages = [{"role": "user", "content": prompt}]
    
    response = openai.ChatCompletion.create(model = model,
                                            messages = messages,
                                            temperature=temperature
                                            )
    return response.choices[0].message["content"] 


# response = get_completion(prompt)
# print(response)
