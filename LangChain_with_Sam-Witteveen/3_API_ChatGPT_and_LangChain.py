"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro

Usando API ChatGPT com LangChain
-------------------------------- 
"""

# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]


import openai

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Você é um assistente útil."},
        {"role": "user", "content": "Olá, que tipo de assistente você é?"},
             ]
)

# Printando a respostas:
#print(response)

# Aqui podemos ver o ROLE:
messages=[
        {"role": "system", "content": "Você é uma assistente prestativa chamada Kate."},
        {"role": "user", "content": "Olá, que tipo de assistente você é?"},
         ]


conversation_total_tokens = 0

while True:
    message = input("Human: ")
    if message=='exit':
        print(f"{conversation_total_tokens} Tokens usados no total nesta conversa.")
        break
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
    
    reply = response.choices[0].message.content
    total_tokens = response.usage['total_tokens']
    conversation_total_tokens += total_tokens
    print(f"ChatGPT: {reply} \n {total_tokens} tokens usados")
    messages.append({"role": "assistant", "content": reply})

"""
Comente o anterior para você usar o seguinte teste 🥸!
"""

"""
ChatGPT com LangChain
---------------------
No terminal pode ver a versão --> $pip show langchain
"""

from langchain import PromptTemplate, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI, OpenAIChat

prefix_messages = [{"role": "system", "content": "Você é uma professora de história prestativa chamada Kate."}]

# Maneira antiga:
#llm = OpenAI(model_name="text-davinci-003",
#             temperature=0, )

# Nova maneira:
llm = OpenAIChat(model_name='gpt-3.5-turbo',
                 temperature=0, 
                 prefix_messages=prefix_messages,
                 max_tokens = 256,
                 verbose=True
                )

# Nosso Prompt Template:
template ="""Faça a seguinte pergunta: {user_input}
           Responda de forma informativa e interessante, mas concisa para alguém que é novo neste tópico.
          """

prompt = PromptTemplate(template=template, 
                        input_variables=["user_input"]
                       )

# A nossa chain:
llm_chain = LLMChain(prompt=prompt,
                     llm=llm
                    )

#user_input = "When was Marcus Aurelius the emperor of Rome?"
#llm_chain.run(user_input)

print("Digite a sua pergunta para começar uma conversa com a AI: ")
while True:
    user_input = input("Human: ")
    result = llm_chain.run(user_input)
    print("AI:", result)

    if not user_input:
        break