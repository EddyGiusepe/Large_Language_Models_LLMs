import spacy
import csv

nlp = spacy.load("pt_core_news_md")

def extrair_entidades(conversa):
    doc = nlp(conversa)
    entidades = []
    for entidade in doc.ents:
        entidades.append((entidade.text, entidade.label_))
    return entidades

conversas = []

while True:
    mensagem = input("Usuário: ")
    if mensagem.lower() == "sair":
        break
    conversas.append(("Usuário", mensagem))
    resposta = "Assistente: Em que posso ajudá-lo?"
    conversas.append(("Assistente", resposta))
    entidades = extrair_entidades(mensagem)
    if entidades:
        with open("entidades.csv", "a") as f:
            writer = csv.writer(f)
            for entidade in entidades:
                writer.writerow([entidade[0], entidade[1]])

print("Obrigado pela conversa!")

