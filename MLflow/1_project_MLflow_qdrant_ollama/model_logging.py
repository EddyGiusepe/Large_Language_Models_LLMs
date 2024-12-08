#!/usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script para logging e servir um modelo RAG (Retrieval-Augmented Generation) usando MLflow e LlamaIndex
======================================================================================================

Este script implementa:
1. Configuração e logging de um modelo RAG no MLflow
2. Carregamento do modelo indexado
3. Interface de queries interativa para o usuário

EXECUÇÃO
--------
Passo 1: Executar ollama e MLflow, no terminal ---> ollama serve    e     mlflow server --host 127.0.0.1 --port 3001
Passo 2: Executar o script, no terminal ---> python model_logging.py  (OBS: Verificar se o arquivo model_indexing.py está no diretório correto)
Passo 3: Interagir com o sistema ---> Digitar as perguntas/queries
"""
from llama_index.core import VectorStoreIndex
import mlflow

# Configuração do MLflow:
mlflow.set_tracking_uri('http://127.0.0.1:3001')
mlflow.set_experiment(experiment_name='llamaindex-qdrant-rag-1')

# Serve para habilitar o registro automático (autologging) de métricas, parâmetros e artefatos relacionados ao LlamaIndex no MLflow:
mlflow.llama_index.autolog() 

def log_and_load_model():
    """
    Registra o modelo no MLflow e carrega o índice vetorial.
    
    Returns:
        VectorStoreIndex: Índice carregado para consultas
    """
    with mlflow.start_run():
        model_info = mlflow.llama_index.log_model("model_indexing.py",
                                                   artifact_path="index",
                                                   engine_type="chat",
                                                 )

    print(f"Model URI: {model_info.model_uri}")

    return mlflow.llama_index.load_model(model_info.model_uri)    # Carregando o índice


def run_query_interface(index: VectorStoreIndex):
    """
    Executa uma interface interativa de consulta para o usuário.
    
    Args:
        index (VectorStoreIndex): Índice vetorial carregado para processar as consultas
    """
    query_engine = index.as_query_engine()
    
    print("🤗 Bem-vindo ao sistema de consultas RAG! 🤗")
    print("Digite 'sair' para encerrar o App")
    
    while True:
        pergunta = input("\nDigite sua pergunta: ")
        
        if pergunta.lower() == 'sair':
            print("Encerrando o programa...")
            break
            
        print("\nProcessando sua pergunta...")
        resposta = query_engine.query(pergunta)
        print(f"\nResposta: {resposta}")





if __name__ == "__main__":
    # Carregamos o modelo e iniciamos a interface de consulta:
    index = log_and_load_model()
    run_query_interface(index)