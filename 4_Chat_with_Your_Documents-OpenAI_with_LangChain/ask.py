"""
Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro

Nota:
     Antes de executar este script, executar primeiro o outro (seguidamente o outro script):
     
     $ python init_vectorstore.py
     $ python ask.py
"""

from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

import os
os.environ["OPENAI_API_KEY"] = "CHAVE API"

embeddings = OpenAIEmbeddings()

docsearch = Chroma(persist_directory="db", embedding_function=embeddings)

chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0),
                                                    chain_type="stuff",
                                                    retriever=docsearch.as_retriever()
                                                   )


# user_input = input("Faça a sua pergunta: ")
# result = chain({"question": user_input}, return_only_outputs=True)
# print("Answer: " + result["answer"].replace('\n', ' '))
# print("Source: " + result["sources"])

print("Digite a sua pergunta para começar uma conversa com a AI: ")
while True:
    user_input = input()
    result = chain({"question": user_input}, return_only_outputs=True)
    print(result["answer"].replace('\n', ' '))

    if not user_input:
        break
