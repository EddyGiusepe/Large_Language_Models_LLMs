"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro

LangChain Chat with Flan20B
---------------------------
Aqui vamos aprender a configurar o Flan20B usando o HuggingFace Hub e,
em seguida, criar uma cadeia de conversação simples com memória e como
rastrear os tokens que estão sendo usados.

!pip -q install huggingface_hub langchain transformers
!pip show langchain
"""

# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]


from langchain.llms import HuggingFaceHub

flan_ul2 = HuggingFaceHub(repo_id="google/flan-ul2", # https://huggingface.co/google/flan-ul2#introduction-to-ul2
                          model_kwargs={"temperature":0.1, "max_new_tokens":150}
                         )

flan_t5 = HuggingFaceHub(repo_id="google/flan-t5-xl",
                         model_kwargs={"temperature":0}
                        )


from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()

conversation = ConversationChain(llm=flan_ul2,
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
#     print("Aqui podemos printar as conversas da memória: ", conversation.memory.buffer)
#     # A seguir podemos carregar toda a HISTÓRIA da conversa:
#     print(memory.load_memory_variables({}))

#     if not query:
#         break


"""
Contando os Tokens
------------------
"""
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

# test_input = "Now Good Morning Ms Rogers"
# # tokenizer([test_input])
# print(tokenizer.tokenize(test_input)) 

# print(conversation.memory.buffer)


# formatted_prompt = conversation.prompt.format(input='the next input', history=memory.buffer)
# print(formatted_prompt)


"""
Verificar antes de usar a seguinte função:
$ pip install torch
$ pip install tensorflow
"""
import torch
print(torch.__version__)


#import tensorflow as tf
#print(tf.__version__)

def chat_to_llm(chat_llm):
    conversation_total_tokens = 0
    new_conversation = ConversationChain(llm=chat_llm, 
                                     verbose=False, 
                                     memory=ConversationBufferMemory()
                                     )
    
    while True:
        message = input("Human: ")
        if message=='exit':
            print(f"{conversation_total_tokens} tokens used in total in this conversation")
            break
        if message:
            formatted_prompt = conversation.prompt.format(input=message,history=new_conversation.memory.buffer)
            num_tokens = len(tokenizer.tokenize(formatted_prompt))
            conversation_total_tokens += num_tokens
            print(f'tokens sent {num_tokens}')
            response = new_conversation.predict(input=message)
            response_num_tokens = len(tokenizer.tokenize(response))
            conversation_total_tokens += response_num_tokens
            print(f"LLM: {response}")

# Aqui você começa a conversa com a AI:
#print(chat_to_llm(flan_ul2)) # Descomenta para usar este Modelo!

# Outra conversa com o outro modelo:
chat_to_llm(flan_t5)


"""
Usando uma memória de resumo
----------------------------
Comente as partes acima para você usar esta memória de resumo.
"""
from langchain.chains.conversation.memory import ConversationSummaryMemory

summary_memory = ConversationSummaryMemory(llm=flan_ul2)

conversation = ConversationChain(llm=flan_ul2,
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
