"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro

🤗 Baseado no tutorial de jcharistech.com 🤗

Resumo de texto com SpaCy
-------------------------
* A sumarização de texto é o processo de destilar as informações mais importantes de uma fonte (ou fontes)
  para produzir uma versão resumida para um determinado usuário (ou usuários) e tarefa (ou tarefas).

* A ideia da sumarização é encontrar um subconjunto de dados que contenha a “informação” de todo o conjunto

Idéia principal:
================
    * Pré-processamento de texto (remover stopwords, pontuações).
    * Tabela de frequência de palavras/Distribuição de frequência de palavras - quantas vezes cada palavra aparece no documento
    * Pontue cada frase dependendo das palavras que ela contém e da tabela de frequência
    * Crie um resumo juntando todas as frases acima de um determinado limite de pontuação   
"""
import spacy
# Pacote de pré-processamento de texto:
from spacy.lang.pt.stop_words import STOP_WORDS
from string import punctuation
# Criando uma list de stopwords:
stopwords = list(STOP_WORDS)
#print(stopwords)
print(len(stopwords))

document1 ="""O aprendizado de máquina (ML) é o estudo científico de algoritmos e modelos estatísticos que os sistemas de computador
usam para melhorar progressivamente seu desempenho em uma tarefa específica. Os algoritmos de aprendizado de máquina constroem um modelo
matemático de dados de amostra, conhecidos como "dados de treinamento", para fazer previsões ou decisões sem serem explicitamente
programados para executar a tarefa. Algoritmos de aprendizado de máquina são utilizados nas aplicações de filtragem de e-mail, detecção
de intrusos de rede e visão computacional, onde é inviável desenvolver um algoritmo de instruções específicas para a execução da tarefa.
O aprendizado de máquina está intimamente relacionado à estatística computacional, que se concentra em fazer previsões usando computadores.
O estudo da otimização matemática fornece métodos, teoria e domínios de aplicação para o campo do aprendizado de máquina. A mineração de
dados é um campo de estudo dentro do aprendizado de máquina e se concentra na análise exploratória de dados por meio do aprendizado não supervisionado.
Em sua aplicação em problemas de negócios, o aprendizado de máquina também é conhecido como análise preditiva.
"""



nlp = spacy.load('pt_core_news_lg')


# Construindo um objeto NLP:
docx = nlp(document1)

# Tokenization do texto:
mytokens = [token.text for token in docx]

"""
Tabela de Frequência de Palavras
--------------------------------

* Dicionário de palavras e suas contagens
* Quantas vezes cada palavra aparece no documento
* Usando non-stopwords

"""
# Construir Frequência de Palavras
# word.text é tokenization no spaCy:
word_frequencies = {}
for word in docx:
    if word.text not in stopwords:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1


#print(word_frequencies)

"""
Frequência Máxima de Palavras
-----------------------------

* Encontre a frequência ponderada
* Cada palavra sobre a palavra que mais ocorre
* Frase longa sobre frase curta

"""
# Frequência Máxima de Palavras
maximum_frequency = max(word_frequencies.values())

for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

"""Distribuição de Frequência de Palavras"""
# Tabela de frequência:
print("")
#print(word_frequencies)


"""
Pontuação da frase e classificação das palavras em cada frase
=============================================================

* Tokens de sentença
* Pontuando cada frase com base no número de palavras
* Sem stopwords em nossa tabela de frequência de palavras
"""
# Sentence Tokens
sentence_list = [ sentence for sentence in docx.sents ]

# Example of Sentence Tokenization,Word Tokenization and Lowering All Text
# for t in sentence_list:
#     for w in t:
#         print(w.text.lower())
[w.text.lower() for t in sentence_list for w in t ]

# Pontuação da sentença comparando cada palavra com a sentença:
sentence_scores = {}  
for sent in sentence_list:  
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if len(sent.text.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]


# Sentence Score via comparrng each word with sentence
# Alternative Method
lowered_sentence_list = [w.text.lower() for t in sentence_list for w in t ]
lowered_sentence_scores = {}  
for sent in lowered_sentence_list:  
        for word in sent.split():
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        lowered_sentence_scores[sent] = word_frequencies[word]
                    else:
                        lowered_sentence_scores[sent] += word_frequencies[word]

"""Obter pontuação da frase"""
# Sentence Score Table
#print(sentence_scores)

"""
Encontrando a Sentença Top N com maior pontuação
================================================
* Usando heapq
"""
# Import Heapq 
from heapq import nlargest

summarized_sentences = nlargest(7, sentence_scores, key=sentence_scores.get)

#print(summarized_sentences)

# Converta sentenças de Spacy Span em Strings para unir a frase inteira:
for w in summarized_sentences:
     w.text
    #print(w.text)

# List comprehension of sentenças convertidas de spacy.span para strings:
final_sentences = [ w.text for w in summarized_sentences ]


"""Juntando as SENTENÇAS"""

summary = ' '.join(final_sentences)
print(summary)

# Comprimento do resumo:
print(len(summary))


# Comprimento do texto original:
print(len(document1))
