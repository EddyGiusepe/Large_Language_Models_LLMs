"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro


Conversation Knowledge Graph Memory
-----------------------------------
Esse tipo de memória usa um GRAFO de CONHECIMENTO para recriar a memória.

Característica chave:
                     A memória do grafo de conhecimento da conversação mantém um
                     grafo de conhecimento de todas as entidades que foram mencionadasnas interações junto com suas relações semânticas.
"""

from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationKGMemory
from langchain.prompts.prompt import PromptTemplate


# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]


# Instanciamos nosso modelo:
llm_model = OpenAI(model_name='text-davinci-003', # Mais recente --> 'gpt-3.5-turbo'. # 'gpt-3.5-turbo-0301' só até 1° de Junho de 2023 
                   temperature=0, 
                   max_tokens = 150
                  )


template = """O que se segue é uma conversa amigável entre um humano e uma IA. A IA é falante e fornece muitos detalhes
específicos de seu contexto. Se a IA não souber a resposta para uma pergunta, ela diz com sinceridade que não sabe.
A IA APENAS usa informações contidas na seção "Informações relevantes" e não alucina.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""

prompt = PromptTemplate(input_variables=["history", "input"],
                        template=template
                       )

memory = ConversationKGMemory(llm=llm_model,
                              return_messages=True
                             )

print(memory.chat_memory)
#print(memory.get_current_entities("Olá Jhon, você é de Portugal?"))
conversation_with_kg = ConversationChain(llm=llm_model,
                                         verbose=True,
                                         prompt=prompt,
                                         memory=memory
                                        )


# Esta pequena parte comentada é para ver toda a estrutura da resposta (xxx)
# query = input("Digite: ")
# xxx = conversation_with_kg({"input": query})
# print(xxx)


print("Digite a sua pergunta para começar uma conversa com a AI: ")
while True:
    query = input("Human: ")
    result = conversation_with_kg({"input": query})
    print("AI: " + result['response'])
    print("")
    print("Mantém um Grafo de conhecimento: ", conversation_with_kg.memory.kg.get_triples())
    print(memory.get_current_entities({"input": query}))
    if not query:
        break
