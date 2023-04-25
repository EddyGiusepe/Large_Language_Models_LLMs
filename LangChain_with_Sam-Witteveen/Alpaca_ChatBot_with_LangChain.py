"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro


Talking to Alpaca with LangChain - Creating an Alpaca ChatBot
-------------------------------------------------------------

Neste script seguiremos o paso paso de Sam Witteveen.
Basicamente usaremos os Alpaca 7B e o LangChain com um Cadeia de conversação
e uma janela de memória.


Instalar:
!pip -q install git+https://github.com/huggingface/transformers 
!pip install -q datasets loralib sentencepiece 
!pip -q install bitsandbytes accelerate
!pip -q install langchain
"""
# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]

"""
Hugging Face
------------
Existem dois wrappers Hugging Face LLM, um para um pipeline local e outro para um
modelo hospedado no Hugging Face Hub. Observe que esses wrappers funcionam apenas
para modelos que suportam as seguintes tarefas: text2text-generation, text-generation

Carregando Alpaca7B:
====================
Lembre que Alpaca foi construida sobre Llama.
"""
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

import torch

tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")

base_model = LlamaForCausalLM.from_pretrained(
    "chavinlo/alpaca-native",
    load_in_8bit=True,
    device_map='auto',
)

pipe = pipeline(
    "text-generation",
    model=base_model, 
    tokenizer=tokenizer, 
    max_length=256,
    temperature=0.0,
    top_p=0.95,
    repetition_penalty=1.2
)

local_llm = HuggingFacePipeline(pipeline=pipe)


from langchain import PromptTemplate, LLMChain

template = """Below is an instruction that describes a task. Write a response that appropriately
              completes the request.

### Instruction: 
{instruction}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["instruction"])


llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                     )


#question = "What is the capital of England?"
#print(llm_chain.run(question))

# Repara que este modelo, Alpaca, apenas interage em Inglês:
# print("Digite a sua pergunta para começar uma conversa com a AI: ")
# while True:
#     user_input = input("Human: ")
#     result = llm_chain.run(user_input)
#     print("AI:", result)

#     if not user_input:
#         break


"""
Configurando um Chat com Memória
--------------------------------
"""
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


# Vamos configurar a memória para voltar 4 turnos:
window_memory = ConversationBufferWindowMemory(k=4)

conversation = ConversationChain(
    llm=local_llm, 
    verbose=True, 
    memory=window_memory
)

print(conversation.prompt.template)

"""
O que se segue é uma conversa amigável entre um humano e uma IA chamada.
A IA é falante e fornece muitos detalhes específicos de seu contexto. Se a IA não
souber a resposta para uma pergunta, ela diz com sinceridade que não sabe.

Conversa atual: {history} 

Humano: {input} AI:
"""

conversation.prompt.template = """The following is a friendly conversation between a
                                  human and an AI called Alpaca. The AI is talkative and
                                  provides lots of specific details from its context.
                                  If the AI does not know the answer to a question, it
                                  truthfully says it does not know. 

Current conversation:
{history}
Human: {input}
AI:"""

# Conversação com Memória:
print("Digite a sua pergunta para começar uma conversa com a AI: ")
while True:
    user_input = input("Human: ")
    result = conversation.predict(input = user_input)
    print("AI:", result)

    if not user_input:
        break
