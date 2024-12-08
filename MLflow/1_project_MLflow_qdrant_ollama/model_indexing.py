#!/usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro


Script para indexação de documentos PDF utilizando LlamaIndex e MLflow
======================================================================

Este script implementa um pipeline de processamento que:
1. Configura um modelo de linguagem usando Ollama
2. Configura embeddings usando OpenAI ou OllamaEmbedding
3. Carrega documentos PDF de um diretório
4. Armazena vetores em um banco Qdrant
5. Cria um índice pesquisável
6. Registra o modelo no MLflow
"""

# Importações necessárias:
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import mlflow
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# Configuração do modelo de linguagem usando Ollama:
Settings.llm = Ollama(model=os.environ.get('LANGUAGE_MODEL'),
                      base_url=os.environ.get('OLLAMA_URL')
                     )

# Configuração do modelo de embeddings usando Ollama (opcional):
# Settings.embed_model = OllamaEmbedding(model_name=os.environ.get('EMBED_MODEL'),
#                                        base_url=os.environ.get('OLLAMA_URL')
#                                       )

# Configuração do modelo de embeddings usando OpenAI:
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",  # Opções disponíveis: "text-embedding-3-large"  ou  "text-embedding-3-small" 
    api_key=os.environ.get('OPENAI_API_KEY')
)

# Configurar o text splitter (antes de criar o índice):
text_splitter = SentenceSplitter(
    chunk_size=8000,        # Número de caracteres por chunk
    chunk_overlap=300,      # Quantidade de sobreposição entre chunks
    separator="\n"       # Separador para dividir o texto
)

# Carregamento dos documentos PDF do diretório ./data:
documents = SimpleDirectoryReader(input_dir='./data',
                                  required_exts=['.pdf']
                                 ).load_data(show_progress=True)

# Inicialização do cliente Qdrant para armazenar os vetores:
client = qdrant_client.QdrantClient(url=os.environ.get('QDRANT_URL'),
                                    api_key=os.environ.get('QDRANT_API_KEY')
                                   )

# Configura o armazenamento dos vetores no Qdrant:
vector_store = QdrantVectorStore(client=client,
                                 collection_name=os.environ.get('COLLECTION_NAME')
                                )

# Cria o contexto de armazenamento para o índice:
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Cria índice de documentos usando o contexto de armazenamento:
index = VectorStoreIndex.from_documents(documents=documents,
                                        storage_context=storage_context,
                                        transformations=[text_splitter],
                                        show_progress=True
                                       )

# Registra o modelo no MLflow para tracking e versionamento:
mlflow.models.set_model(index)
