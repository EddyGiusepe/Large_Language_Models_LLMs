"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro


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

# document1 ="""O aprendizado de máquina (ML) é o estudo científico de algoritmos e modelos estatísticos que os sistemas de computador
# usam para melhorar progressivamente seu desempenho em uma tarefa específica. Os algoritmos de aprendizado de máquina constroem um modelo
# matemático de dados de amostra, conhecidos como "dados de treinamento", para fazer previsões ou decisões sem serem explicitamente
# programados para executar a tarefa. Algoritmos de aprendizado de máquina são utilizados nas aplicações de filtragem de e-mail, detecção
# de intrusos de rede e visão computacional, onde é inviável desenvolver um algoritmo de instruções específicas para a execução da tarefa.
# O aprendizado de máquina está intimamente relacionado à estatística computacional, que se concentra em fazer previsões usando computadores.
# O estudo da otimização matemática fornece métodos, teoria e domínios de aplicação para o campo do aprendizado de máquina. A mineração de
# dados é um campo de estudo dentro do aprendizado de máquina e se concentra na análise exploratória de dados por meio do aprendizado não supervisionado.
# Em sua aplicação em problemas de negócios, o aprendizado de máquina também é conhecido como análise preditiva.
# """

document1 = """
  orientações para o agendamento – sedes clique aqui  gdf disponibiliza agências do brb para agendamento nos cras – dias 03 e 04 de janeiro/2023 ♦    atualizado em 24/02/2023  - por motivo de força maior, o cras paranoá teve que passar por manutenção, portanto, não está realizando atendimento na modalidade presencial. nesse sentido, não houve a suspensão do atendimento no cras paranoá, mas sim a necessidade de mudança na modalidade do atendimento; logo, os usuários serão atendidos no dia e horário agendado, mas de forma remota. atenção: destaca-se que se trata de uma situação temporária.  atualizado em 31/01/2023 - informamos que o  cras arapoanga se encontra em reforma, por isso, os atendimentos que estavam agendados no formato presencial serão atendidos remotamente, ou seja, por telefone. portanto, o usuário não será prejudicado. o usuário deverá aguardar o contato do servidor do cras, que fará o atendimento por telefone.      novo sistema de agendamento cras a partir do dia 07 de dezembro de 2022, estará disponível o novo sistema de agendamento cras. isso significa que 100% das vagas de atendimento inicial nos cras serão agendadas por meio do site da sedes ou por contato com a central 156.  segue na íntegra o manual de operação do novo sistema de agendamento:  clique aqui    o operador poderá proceder normalmente com o agendamento, caso haja vaga, ou seja, deverá realizar o agendamento de acordo com o funcionamento da central 156 (das 07h00 às 21h00; e aos sábados, domingos e feriados, das 08h às 18h);  caso o cidadão informe que ele ou alguém de sua família foi atendido pelo cras nos últimos 3 meses, não precisa de novo agendamento, basta orientar a procurar a recepção do cras da sua região (abrangência), uma vez que, nesse caso, a família pode obter informação sobre os desdobramentos do seu atendimento;  destaca-se que as vagas de agendamento não serão concorrentes; ou seja, determinada quantidades de vagas serão destinadas para o agendamento via 156 com o atendimento humano (receptivo) para facilitar o acesso das pessoas que não conseguem agendar pela internet.    em relação ao sistema sids (https://sids.sedes.df.gov.br/sistema), permanece a utilização somente para realização de consulta.        atenção:  para as solicitações a seguir não há necessidade de agendamento, devendo o interessado comparecer diretamente a uma das unidades de cras: → atendimentos relacionados a bpc; → carteirinha do idoso; → auxílio por morte; → auxílio natalidade; → declaração de isenção para 2ª via de rg; → auxílio na obtenção de documentação civil.  endereços/abrangências dos cras - centro de referência de assistência social:  clique aqui   endereço cras equipe de proteção social móvel presencial - setor comercial norte, quadra 01,lote g ed rossi esplanada business, loja 01, próximo ao hospital regional de asa norte (hran), brasília.   posto de cadastro único do plano piloto  endereço: srtvs q.701 conjunto l bloco 1 n° 38 sala 601 edifício assis chateaubriand - w3 sul, entre o pátio brasil e a igreja dom bosco, e de frente o hospital sarah.  documentação:   documentos obrigatórios para o responsável familiar (rf): - cpf ou título de eleitor do responsável pela família e documento de identificação com foto (rg, carteira de trabalho, cnh). documentos para membros da família (obrigatório no mínimo um documento de identificação por membro): - certidão de nascimento; - certidão de casamento; - registro geral de identificação (rg); - cadastro de pessoas físicas (cpf); - carteira de trabalho e previdência social; ou - título de eleitor. outros documentos não obrigatórios, mas que facilitam o cadastramento: - comprovante de residência com preferência para as contas de luz e água ,caso não tenha, conta de celular, telefone ou cartão de crédito; - comprovante de rendimentos (carteira de trabalho ou contracheque); - comprovante de matrícula de crianças e adolescentes até 17 anos. se não tiver o comprovante, o responsável familiar deve informar o nome da escola o criança/jovem. atenção: os documentos devem ser originais, uma vez que a unidade não retém cópia da documentação apresentada.  atualizado em 02/01/2023 - gdf disponibiliza agências do brb para agendamento nos cras – dias 03 e 04 de janeiro/2023 nos dias 03 e 04 de janeiro de 2023, a sedes realizará agendamentos aos serviços socioassistenciais/cadastro único em 17 agências do banco de brasília (brb). o público interessado vai ser recebido nas unidades das 8h às 10h. o objetivo é agilizar a marcação de atendimentos, previstos para ocorrerem ao longo de janeiro e fevereiro nos centros de referência de assistência social (cras) e postos de cadastro único. confira as unidades do brb e os respectivos endereços: brb ceilândia norte qnn 25, conjunto c, lote 2/4, ceilândia norte brb ceilândia cnm 1, bloco b, ceilândia centro brb samambaia qn 206, lote 1, conjunto a brb paranoá praça central, área especial, lote 2 brb santa maria quadra central 01, lote 10, bloco b brb planaltina shd bloco j/a – setor comercial central brb vila buritis quadra 02, conjunto b, lote st, residencial leste, vila buritis brb são sebastião centro de múltiplas atividades, lote 6, centro brb recanto das emas quadra 203, lote 15, avenida recanto das emas brb taguatinga scc c8, setor comercial central, lotes 13/14 e 29/30, taguatinga norte brb taguatinga norte cng 4, lotes 17/18, taguatinga norte brb taguatinga sul csd 06, lote 24, taguatinga sul brb brazlândia quadra 3, bloco b, lotes 6/10, setor norte brb gama scc bloco 1, lotes 1/19 brb sobradinho ii avenida central, conj. 6 brb riacho fundo ac 03, lote 10 brb estrutural st área especial scia         atualizado em 02/01/2023, às 07h12.  
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
                if len(sent.text.split(' ')) < 80:
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
                if len(sent.split(' ')) < 80:
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
