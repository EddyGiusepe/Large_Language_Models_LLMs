"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro

Construindo a nossa função para RESUMO 
--------------------------------------
Aqui basicamente usamos spaCy para fazer resumo de nossos Textos.
"""

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


import spacy
from heapq import nlargest
from spacy.lang.pt.stop_words import STOP_WORDS
from string import punctuation
stopwords = list(STOP_WORDS)

nlp = spacy.load('pt_core_news_lg')


def text_summarizer(raw_docx):
    raw_text = raw_docx
    docx = nlp(raw_text)
    stopwords = list(STOP_WORDS)
    # Construindo Frequência de Palavras.
    # word.text é tokenization no spaCy:
    word_frequencies = {}  
    for word in docx:  
        if word.text not in stopwords:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1


    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    # Toneks de sentenças
    sentence_list = [ sentence for sentence in docx.sents ]

    # Calcular pontuação e classificação da sentença:
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if len(sent.text.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]

    # Encontrar N largest:
    summary_sentences = nlargest(10, sentence_scores, key=sentence_scores.get)
    final_sentences = [ w.text for w in summary_sentences ]
    summary = ' '.join(final_sentences)
    print("Documento original\n")
    print(raw_docx)
    print("Comprimento total do texto original:",len(raw_docx))
    print('\n\nDocumento Resumido\n')
    print(summary)
    print("\nComprimento total do texto resumido:",len(summary))


# Testamos com algum documento, assim:
text_summarizer(document1)
