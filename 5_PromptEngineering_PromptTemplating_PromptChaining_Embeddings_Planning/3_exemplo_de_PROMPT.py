# Isto é quando usas o arquivo .env:
import openai
import spacy 
from dotenv import load_dotenv
import os

print('Carregando a Chave: ', load_dotenv())
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY']  

# Adicione isto para usar a sua have:
openai.api_key = Eddy_API_KEY_OpenAI


# Carregue o modelo da linguagem do spaCy para português
nlp = spacy.load("pt_core_news_md")

# Função para extrair entidades nomeadas de uma frase
def extrair_entidades(frase):
    doc = nlp(frase)
    return [(ent.text, ent.label_) for ent in doc.ents]


# Loop da conversa
while True:
    # Exibe o prompt e aguarda a entrada do usuário
    user_input = input("Você: ")
    
    # Extrai as entidades nomeadas da entrada do usuário
    entidades = extrair_entidades(user_input)
    
    print(entidades)
    # Cria o prompt com as entidades extraídas
    prompt = f"""
    Você deve atuar como um assistente da COCRIAR. A sua tarefa é ajudar a fornecer informações ao 
    usuário que interage com você. O usuário pede informações sobre diversos tópicos, por exemplo: organizações públicas ou privadas,
    endereços das mesmas, procedimentos para vacinação, serviço público em geral, etc. 
    Você deve responder de forma concisa e clara. 
    
    Você deve levar em conta as Entidades que o usuário menciona na conversa ou na pergunta e deve lembrar delas para depois 
    fornecer uma resposta, com contexto, em base a essa entidade ou entidades mencionada na conversa. 
    
    Se você não souber a resposta, você responde com "Não sei a resposta". 

    Na sua resposta use no máximo 20 palavras, focando em aspectos relevantes que
    enriqueçam a sua resposta dentro do contexto da conversa.
    
    Usuário: {user_input}
    
    Entidades: {entidades}
    
    Assistente da COCRIAR:
    """
    
    # Gera a resposta do assistente da COCRIAR
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.0,
    )
    
    # Exibe a resposta do assistente de AI
    #print(response)
    print("🤗")
    resposta = response.choices[0].message["content"].strip()
    print("Assistente da COCRIAR:", resposta)



