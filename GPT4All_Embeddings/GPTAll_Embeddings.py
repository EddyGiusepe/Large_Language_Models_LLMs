"""
Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro

Embeddings
==========
O GPT4All oferece suporte à geração de EMBEDDINGS de alta qualidade de 
documentos de texto de tamanho arbitrário usando um Transformador de Sentença 
treinado de forma CONTRASTIVA otimizado para CPU. Esses Embeddings são comparáveis 
em qualidade para muitas tarefas com OpenAI.

Para mais detalhes, ver o seguinte link:

* https://docs.gpt4all.io/gpt4all_python_embedding.html
"""
from gpt4all import GPT4All, Embed4All
text = 'A OpenAI tem sede em São Francisco.'
embedder = Embed4All()
output = embedder.embed(text)
print("")
print(output)
print("")
print(len(output))