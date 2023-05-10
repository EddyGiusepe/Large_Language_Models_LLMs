"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro

Este trabalho automatizado foi baseado no maravilhoso trabalho de Hasan Aboul Hasan 🤗

ChatGPT e o uso das correntes
=============================

Nesta pasta temos:

1. Prompts Avançados
   -----------------
   
   No script prompts.py, encontrará uma lista de prompts. Você pode customizar!

2. Automação 🤖
   ---------

   Você não precisa ir toda vez e pedir ao ChatGPT cada tarefa manualmente. Tudo é automatizado em 1 clique!

3. Encadeamento
   ------------

   No script app.py você observará essa estrutura de encadeamento.
   Este script é a parte central.  Você deve estar observando que estamos encadeando as saídas.
   Por exemplo, a saída da primeira tarefa (Ideias de Títulos) é a entrada da segunda (Ideias de Miniaturas).
   Essa ideia é muito importante pois estamos alimentando automaticamente os resultados e construindo em cima disso 🥳!


Você pode executar das seguintes maneiras:

* No terminal, você executará o arquivo shell --> $ . ./run-me.sh

ou

* Via python mesmo --> $ python app.py
"""

import chat_gpt_api as gpt
import prompts as pr


#step 1: Insira um tópico
user_topic = input("Insira o tópico do seu vídeo: ")
user_minutes = input("Insira a duração do seu vídeo (em minutos): ")

#step 2: Gere 10 ideias de títulos cativantes
titles_prompt = pr.youtube_title_generator_prompt.format(topic=user_topic)
titles = gpt.basic_generation(titles_prompt)
print("Ideias de Títulos: ")
print("----------------")
print(titles)
print("----------------")


#step 3: Gere ideias atraentes para miniaturas
thumbnail_prompt = pr.youtube_thumbmail_generator_prompt.format(user_titles=titles)
thumbnails = gpt.basic_generation(thumbnail_prompt)
print("Ideias para Miniaturas: ")
print("----------------")
print(thumbnails)
print("----------------")

#step 4: Roteiro
script_prompt = pr.youtube_script_generator_prompt.format(minutes=user_minutes,topic=user_topic)
script = gpt.basic_generation(script_prompt)
print("Sugestão de roteiro: ")
print("----------------")
print(script)
print("----------------")



#step 5: Em um tópico do twitter
tweet_prompt = pr.tweet_from_youtube_prompt.format(youtube_transcript=script)
tweet = gpt.basic_generation(tweet_prompt)
print("Tópico do Twitter: ")
print("----------------")
print(tweet)
print("----------------")