"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro

Usando modelos de HuggingFace localmente
----------------------------------------
Aqui vamos aprender a carregar modelos de HuggingFace localmente para que você
possa usar modelos que não pode usar por meio dos endpoints da API.

!pip -q install langchain huggingface_hub transformers sentence_transformers
!pip show langchain
"""

"""
HuggingFace - HuggingFaceHub
----------------------------
Existem dois wrappers Hugging Face LLM, um para um pipeline local e outro para
um modelo hospedado no Hugging Face Hub. Observe que esses wrappers funcionam apenas
para modelos que suportam as seguintes tarefas: text2text-generation, text-generation
"""
# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]


from langchain import PromptTemplate, HuggingFaceHub, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template,
                        input_variables=["question"])

# Funciona bem para o Inglês. Mas não é conversacional!  
llm_chain = LLMChain(prompt=prompt, 
                     llm=HuggingFaceHub(repo_id="google/flan-t5-xl", 
                                        model_kwargs={"temperature":0, 
                                                      "max_length":64}))


#user_input = "When was Marcus Aurelius the emperor of Rome?"
#llm_chain.run(user_input)

# print("Digite a sua pergunta para começar uma conversa com a AI: ")
# while True:
#     user_input = input("Human: ")
#     result = llm_chain.run(user_input)
#     print("AI:", result)

#     if not user_input:
#         break



"""
BlenderBot
----------
Foi criado pelo Facebook especificamente para bate-papo.
Não funciona no hub 🥹.
"""
# blenderbot_chain = LLMChain(prompt=prompt, 
#                      llm=HuggingFaceHub(repo_id="facebook/blenderbot-1B-distill", 
#                                         model_kwargs={"temperature":0, 
#                                                       "max_length":64}))



"""
Modelos de HF localmente
------------------------
Por que você deseja usar o modo local?
* modelos ajustados (fine-tuned)
* GPU hospedado etc
* alguns modelos só funcionam localmente

Precisei instalar:
$ pip install accelerate
$ pip install bitsandbytes
"""

"""
T5-Flan - Encoder-Decoder
-------------------------
"""
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import bitsandbytes

model_id = 'google/flan-t5-large' # Escolha um modelo pequeno se em caso não tiver VRAM
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForSeq2SeqLM.from_pretrained(model_id, from_tf=True
                                             )


pipe = pipeline(task="text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=100
               )

local_llm = HuggingFacePipeline(pipeline=pipe,
                                verbose=True
                               )

print("")
print("Fazemos uma primeira pergunta: ")
print(local_llm('What is the capital of France? '))


# Usamos LangChain:
llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                     )

#question = "What is the capital of England?"
#print(llm_chain.run(question))

# print("Digite a sua pergunta para a AI: ")
# while True:
#     user_input = input("Human: ")
#     result = llm_chain.run(user_input)
#     print("AI:", result)

#     if not user_input:
#         break


"""
GPT2-medium - Modelo Somente Decodificador
------------------------------------------
microsoft/DialoGPT-large

Aqui vamos gerar texto.
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_id = "gpt2-medium"
#tokenizer = AutoTokenizer.from_pretrained(model_id)
#model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)

pipe = pipeline("text-generation",
                model=model, 
                tokenizer=tokenizer, 
                max_length=100
               )

local_llm = HuggingFacePipeline(pipeline=pipe)

llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                    )

# print("Digite a sua pergunta para a AI: ")
# while True:
#     user_input = input("Human: ")
#     result = llm_chain.run(user_input)
#     print("AI:", result)

#     if not user_input:
#         break


"""
BlenderBot - Codificador-Decodificador
--------------------------------------
"""
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

model_id = 'facebook/blenderbot-1B-distill'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(task="text2text-generation",
                model=model, 
                tokenizer=tokenizer, 
                max_length=100
               )

local_llm = HuggingFacePipeline(pipeline=pipe)

llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                     )


# print("Digite a sua pergunta para a AI: ")
# while True:
#     user_input = input("Human: ")
#     result = llm_chain.run(user_input)
#     print("AI:", result)

#     if not user_input:
#         break



"""
SentenceTransformers
--------------------
"""
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceHubEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"

hf = HuggingFaceEmbeddings(model_name=model_name)


print("Calculamos EMBEDDINGS de uma string: ", hf.embed_query('this is an embedding'))
print("")
print("EMBEDDINGS de várias strings: ", hf.embed_documents(['this is an embedding','this another embedding']))



# De outra forma:
hf = HuggingFaceHubEmbeddings(repo_id=model_name,
                              task="feature-extraction",
                              # huggingfacehub_api_token="my-api-key",
                              )

print("Calculamos EMBEDDINGS de uma string: ", hf.embed_query('this is an embedding'))
