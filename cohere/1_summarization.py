"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro

cohere (https://docs.cohere.com/docs/the-cohere-platform)
------

Cohere oferece uma API para adicionar processamento de linguagem de ponta a qualquer sistema.
O Cohere treina modelos de linguagem massivos e os coloca atrás de uma API simples. Além disso,
por meio do treinamento, os usuários podem criar modelos massivos personalizados para seu caso
de uso e treinados em seus dados. Dessa forma, o Cohere lida com as complexidades da coleta de
grandes quantidades de dados de texto, das arquiteturas de rede neural em constante evolução,
do treinamento distribuído e dos modelos de serviço 24 horas por dia.

Duas categorias principais de grandes modelos de linguagem são modelos de linguagem generativa
(como GPT2 e GPT3) e modelos de linguagem de representação (como BERT). Cohere oferece variantes
de ambos os tipos.

Links:

* https://www.youtube.com/watch?v=zAFomREYYVs
* https://colab.research.google.com/drive/1ZkCiVh5k1XejLsTLAERGIqEn0szPW8EE?usp=sharing#scrollTo=iTCRvKVtXE1v
"""

# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY']  
Eddy_API_KEY_Cohere = os.environ["COHERE_API_KEY"]
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]
Eddy_API_KEY_SerpApi = os.environ["SERPAPI_API_KEY"]
Eddy_API_KEY_WolframAlpha = os.environ["WOLFRAM_ALPHA_APPID"]

import cohere

# Crie uma conexão com a API Cohere usando sua chave de trilha (trail) 
co = cohere.Client(Eddy_API_KEY_Cohere)


"""
Generation Playground:
======================
"""
# response = co.generate(
#   model='xlarge',
#   prompt='Descreve para mim o que é a Inteligência Artificial (AI):',
#   max_tokens=100,
#   temperature=0.0,
# )

# print(response.generations[0].text)


"""
Summarizing:
============
"""
prompt = '''
  orientações para o agendamento – sedes clique aqui  gdf disponibiliza agências do brb para agendamento
  nos cras – dias 03 e 04 de janeiro/2023 ♦    atualizado em 24/02/2023  - por motivo de força maior, o
  cras paranoá teve que passar por manutenção, portanto, não está realizando atendimento na modalidade
  presencial. nesse sentido, não houve a suspensão do atendimento no cras paranoá, mas sim a necessidade
  de mudança na modalidade do atendimento; logo, os usuários serão atendidos no dia e horário agendado,
  mas de forma remota. atenção: destaca-se que se trata de uma situação temporária.  atualizado
  em 31/01/2023 - informamos que o  cras arapoanga se encontra em reforma, por isso, os atendimentos
  que estavam agendados no formato presencial serão atendidos remotamente, ou seja, por telefone.
  portanto, o usuário não será prejudicado. o usuário deverá aguardar o contato do servidor do cras,
  que fará o atendimento por telefone.      novo sistema de agendamento cras a partir do dia 07 de 
  dezembro de 2022, estará disponível o novo sistema de agendamento cras. isso significa que 100% das 
  vagas de atendimento inicial nos cras serão agendadas por meio do site da sedes ou por contato com a 
  central 156.  segue na íntegra o manual de operação do novo sistema de agendamento:  clique aqui    
  o operador poderá proceder normalmente com o agendamento, caso haja vaga, ou seja, deverá realizar 
  o agendamento de acordo com o funcionamento da central 156 (das 07h00 às 21h00; e aos sábados, 
  domingos e feriados, das 08h às 18h);  caso o cidadão informe que ele ou alguém de sua família foi 
  atendido pelo cras nos últimos 3 meses, não precisa de novo agendamento, basta orientar a procurar 
  a recepção do cras da sua região (abrangência), uma vez que, nesse caso, a família pode obter 
  informação sobre os desdobramentos do seu atendimento;  destaca-se que as vagas de agendamento não 
  serão concorrentes; ou seja, determinada quantidades de vagas serão destinadas para o agendamento via 
  156 com o atendimento humano (receptivo) para facilitar o acesso das pessoas que não conseguem agendar 
  pela internet.    em relação ao sistema sids (https://sids.sedes.df.gov.br/sistema), permanece a 
  utilização somente para realização de consulta.
'''

# response = co.generate(
#   model='xlarge',
#   prompt=f'{prompt}\n Resumo:',
#   max_tokens=200,
#   temperature=0,
# )

# print(response.generations[0].text)
response = co.summarize(text=prompt,
                        model="summarize-xlarge",
                        temperature=0,
                        length='long',
                        format='bullets',
                        extractiveness='medium'
                       )
print(response)