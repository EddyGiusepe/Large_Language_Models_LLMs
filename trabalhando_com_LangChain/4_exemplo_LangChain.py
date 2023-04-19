"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro

A Ideia: 
        Aqui vamos a usar a Memória de Conversação.
        Especificamente usaremos esta Cadeia para ter uma conversa e carregar o contexto da memória.
        https://python.langchain.com/en/latest/reference/modules/chains.html?highlight=ConversationChain#langchain.chains.ConversationChain
"""

# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]


"""
Memory
------
Memória é o conceito de estado persistente entre chamadas de uma cadeia/agente.
LangChain fornece uma interface padrão para memória, uma coleção de implementações
de memória e exemplos de cadeias/agentes que usam memória.
Esta cadeia é usada nas conversas para CARREGAR O CONTEXTO DA MEMÓRIA.
"""
from langchain import OpenAI, ConversationChain
from langchain.memory import ConversationBufferMemory

llm = OpenAI(temperature=0,
             max_tokens=150,
             verbose=True
            )

conversation = ConversationChain(llm=llm,
                                 verbose=True,
                                 memory=ConversationBufferMemory()
                                )

# print("Digite a sua pergunta para começar uma conversa com a AI: ")
# while True:
#     user_input = input()
#     result = conversation.predict(input=user_input)
#     print(result)

#     if not user_input:
#         break
print("Digite a sua pergunta para começar uma conversa com a AI: ")
while True:
    query = input("Human: ")
    result = conversation({"input": query})
    print("AI: " + result['response'])
    
    if not query:
        break
