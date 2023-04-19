"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro

A ideia:
        Aqui conversamos com o nosso documento (no caso um .txt). Basicamente respondemos perguntas com o nosso 
        documento. Usamos `RetrievalQA` com tipo de cadeia 'stuff'

Nota: Na hora de gerar o .txt em formato ascii omitiu os acentos. Verificar isso 🧐!!!       
"""
from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

# Deve instalar --> pip install chromadb
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
#from langchain import OpenAI, VectorDBQA # <-- Deprecate
from langchain.chains import RetrievalQA # <-- Alternativa de uso
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain import OpenAI

# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]


# Diretório persistente para o chromadb, de modo que o armazenamento de vetores seja armazenado localmente.
persist_directory = 'db'
embeddings = OpenAIEmbeddings()

# Verificamos se o DB já existe:
if not os.path.exists(persist_directory):
    # Carregar Newtonian_Physics.txt codificar e decodificar em ascii e gravar no arquivo para evitar erros de codificação
    with open("./Newtonian_Physics.txt", "r", encoding="utf-8") as f:
        book = f.read().encode("ascii", "ignore").decode("ascii")
    with open("./book_ascii.txt", "w") as f:
        f.write(book)        

    print("Carregando book_ascii.txt")
    loader = TextLoader("./Newtonian_Physics.txt")

    documents = loader.load()
    #print(documents[0].page_content[:])

    text_splitter = CharacterTextSplitter(chunk_size=60,
                                          chunk_overlap=0
                                         )
    docs = text_splitter.split_documents(documents)
    #print(docs[0].page_content[:])

    print("embeddings book_ascii.txt")
    db = Chroma.from_documents(docs,
                                embeddings,
                                persist_directory=persist_directory
                                )
else:    
    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings
               )

# Prompt customizado:
prompt_template = """Use as seguintes partes do contexto para responder à pergunta no final, resumindo o
                     contexto em no máximo três pontos relevantes. Se você não sabe a resposta, apenas diga
                     que não sabe, não tente inventar uma resposta.
                  

{context}

Question: {question}
Answer:
"""

PROMPT = PromptTemplate(template=prompt_template,
                         input_variables=["context", "question"])


chain_type_kwargs = {"prompt":PROMPT}

"""
Aqui estamos usando o vectorstore como o banco de dados e não os documentos pesquisados por similaridade,
pois isso é feito na cadeia.
"""
OpenAI_llm = OpenAI(temperature=0.0,
                    verbose=True,
                    max_tokens=150
                   )
# Exponha este índice em uma interface de recuperação:
#retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":1})

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 2

qa = RetrievalQA.from_chain_type(llm=OpenAI_llm,
                                chain_type="stuff",
                                return_source_documents=True, # Tem que ser True para poder printar abaixo --> print(result['source_documents'])
                                retriever=retriever,
                                chain_type_kwargs=chain_type_kwargs
                               )


# while True:
#     query = input("Digite a sua query: ")
#     result = qa(query)
#     print(result['result'])

while True:
    # Para rastrear os Tokens:
    with get_openai_callback() as cb:
        query = input("Digite a sua query: ")
        result = qa(query)
        print(result['result'])
        print(result['source_documents'])
        print("Tokens usados: ", cb.total_tokens)
