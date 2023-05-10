youtube_title_generator_prompt = """\

Quero que você atue como um criador de título viral do YouTube. 
Pense em títulos cativantes e que chamem a atenção e incentivem as pessoas 
a clicar e assistir ao vídeo. Os títulos devem ser curtos, concisos e diretos. 
Eles também devem ser criativos e inteligentes. Tente criar títulos inesperados 
e surpreendentes. Não use títulos muito genéricos ou títulos que já foram usados 
muitas vezes antes. Meu vídeo é sobre {topic}. """


youtube_thumbmail_generator_prompt = """Quero que você atue como um criador de miniaturas viral do 
YouTube. Pense em miniaturas cativantes e que chamem a atenção e incentivem as pessoas a clicar e 
assistir ao vídeo. Vou fornecer a você 10 títulos, e você vai sugerir miniaturas para cada um, 
descreva muito bem o que está na miniatura e seja o mais detalhado possível, para que os designers 
possam entender e criar. Aqui estão os títulos {user_titles}."""


youtube_script_generator_prompt = """Atue como um escritor profissional de roteiros de vídeo do YouTube 
e crie um roteiro envolvente para um vídeo de {minutes} minutos.
Pense fora da caixa e crie um roteiro criativo, espirituoso e cativante que as 
pessoas estariam interessadas em assistir e compartilhar. Utilize técnicas para gerar 
mais engajamento no roteiro da narração. 
Crie uma linha de tempo e cumpra-a por até {minutes} minutos de narração falada.

O Tópico É: [{topic}]"""


tweet_from_youtube_prompt = """Aja como se você fosse um especialista em mídia social. 
Dê-me um tópico de 10 tweets com base na seguinte transcrição do youtube: {youtube_transcript}. 
O tópico deve ser otimizado para viralidade e contém hashtags e emoticons. 
Cada tweet não deve exceder 280 caracteres."""