"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro


Conversas com Memória - LangChain
-----------------------------------
Por que Memória é Importante?

Para que o ChatBot/Agente tenha informações de conversas anteriores e assim poder 
identificar (através das interações anteriores) a certo usuário.
"""


# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]

"""
Conversação básica com ConversationBufferMemory
-----------------------------------------------
"""
from langchain import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 120
            )

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=memory
    )



# print("Digite a sua pergunta para começar uma conversa com a AI: ")
# while True:
#     query = input("Human: ")
#     result = conversation({"input": query})
#     print("AI: " + result['response'])
#     print("")
#     memory.save_context
#     print("")
#     #print("Aqui podemos printar as conversas da memória: ", conversation.memory.buffer)
#     # A seguir podemos carregar toda a HISTÓRIA da conversa:
#     print(memory.load_memory_variables({}))

#     if not query:
#         break



"""
Conversação básica com ConversationSummaryMemory
------------------------------------------------
Obs: Aqui a AI quando faz o resumo gasta mais Tokens com respeito da memória anterior.
"""
from langchain import OpenAI
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain

llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 120
            )

summary_memory = ConversationSummaryMemory(llm=OpenAI()
                                          )
conversation = ConversationChain(llm=llm,
                                 verbose=True,
                                 memory=summary_memory
                                )


# print("Digite a sua pergunta para começar uma conversa com a AI: ")
# while True:
#     query = input("Human: ")
#     result = conversation.predict(input=query)
#     print("AI: " + result)
#     print("")
#     #memory.save_context
#     print("")
#     print("Aqui podemos printar as conversas da memória: ", conversation.memory.buffer)
#     # A seguir podemos carregar toda a HISTÓRIA da conversa:
#     #print(memory.load_memory_variables({}))

#     if not query:
#         break


"""
Conversação básica com ConversationBufferWindowMemory
-----------------------------------------------------
"""
from langchain import OpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 60
            )

# Definimos um k = 2 baixo, para manter apenas as 2 últimas interações na memória:
window_memory = ConversationBufferWindowMemory(k=2)

conversation = ConversationChain(llm=llm,
                                 verbose=True,
                                 memory=window_memory
                                )


# print("Digite a sua pergunta para começar uma conversa com a AI: ")
# while True:
#     query = input("Human: ")
#     result = conversation.predict(input=query)
#     print("AI: " + result)
#     print("")
#     #memory.save_context
#     print("")
#     print("Aqui podemos printar as conversas da memória: ", conversation.memory.buffer)
#     # A seguir podemos carregar toda a HISTÓRIA da conversa:
#     #print(memory.load_memory_variables({}))

#     if not query:
#         break


"""
Conversação básica com ConversationSummaryBufferMemory
------------------------------------------------------
"Ele mantém um buffer de interações recentes na memória,
mas em vez de apenas liberar completamente as interações antigas,
ele as compila em um resumo e usa ambos"
"""
from langchain import OpenAI
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 100
            )


# Definindo k=2, manterá apenas as 2 últimas interações na memória
# max_token_limit=40 - limites de token precisam da instalação de transformers
memory = ConversationSummaryBufferMemory(llm=OpenAI(),
                                         max_token_limit=40
                                        ) 

conversation_with_summary = ConversationChain(llm=llm,
                                              memory=memory,
                                              verbose=True
                                             )

# print("Digite a sua pergunta para começar uma conversa com a AI: ")
# while True:
#     query = input("Human: ")
#     result = conversation_with_summary.predict(input=query)
#     print("AI: " + result)
#     print("")
#     #memory.save_context
#     print("")
#     print("Aqui podemos printar as conversas da memória: ", conversation_with_summary.memory.buffer)
#     # Printamos:
#     print(conversation_with_summary.memory.moving_summary_buffer)
#     # A seguir podemos carregar toda a HISTÓRIA da conversa:
#     #print(memory.load_memory_variables({}))

#     if not query:
#         break


"""
Conversação básica com Conversation Knowledge Graph Memory
----------------------------------------------------------
"""
from langchain import OpenAI
from langchain.chains.conversation.memory import ConversationKGMemory
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate

llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 256
            )


template = """O que se segue é uma conversa amigável, na lingua português do Brasil, entre um humano e uma IA.
              A IA é falante e fornece muitos detalhes específicos de seu contexto. Se a IA não souber a
              resposta para uma pergunta, ela diz com sinceridade que não sabe. A IA APENAS usa informações
              contidas na seção "Informações relevantes" e não alucina.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""

prompt = PromptTemplate(input_variables=["history", "input"],
                        template=template
                       )

memory=ConversationKGMemory(llm=llm,
                            return_messages=True
                           )


conversation_with_kg = ConversationChain(llm=llm,
                                         verbose=True,
                                         prompt=prompt,
                                         memory=memory
                                        )


print("Digite a sua pergunta para começar uma conversa com a AI: ")
while True:
    query = input("Human: ")
    result = conversation_with_kg.predict(input=query)
    print("AI: " + result)
    print("")
    #memory.save_context
    print("")
    #print("Aqui podemos printar as conversas da memória: ", conversation_with_kg.memory)
    # Printamos as Entidades:
    print(memory.get_current_entities(query))
    # A seguir podemos carregar toda a HISTÓRIA da conversa:
    #print(memory.load_memory_variables({}))
    # Printamos os TRIGÊMEOS:
    print(memory.get_knowledge_triplets(query))
    print("")
    print(conversation_with_kg.memory.kg)
    print(conversation_with_kg.memory.kg.get_triples())


    if not query:
        break



