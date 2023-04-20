"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro


Bate-papo com memória
---------------------
Começamos com a memória --> ConversationSummaryMemory()
"""
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from pprint import pprint


# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# Nosso modelo:
llm = OpenAI(temperature=0.0,
             verbose=True,
             max_tokens=150
            )

# Memória de resumo:
conversation = ConversationChain(llm=llm,
                                 verbose=True,
                                 memory=ConversationSummaryMemory(llm=llm,
                                                                  return_messages=True) # Pode sim adicionar --> prompt 
                                )

"""
Abaixo digite a palavra --> "memória" para ligar e desligar a impressão de memória.
"""
while True:
    user_input = input("you: ")
    if user_input == "memória" and conversation.verbose == False:
        conversation.verbose = True
        continue
    elif user_input == "memória" and conversation.verbose == True:
        conversation.verbose = False
        continue
    ai = conversation.predict(input=user_input) # ou use: ai = conversation.run(user_input) 
    print("AI: ", ai)