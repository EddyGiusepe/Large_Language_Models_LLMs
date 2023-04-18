"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro

Esta série de files .py foi baseado no maravilhoso trabalho do Sam Witteveen.
"""


# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]


"""
Geração condicional simples
---------------------------
* Com OpenAI GPT3
"""
from langchain.llms import OpenAI

llm = OpenAI(model_name='text-davinci-003', 
             temperature=0.0, 
             max_tokens = 256
            )


# # Fazemos interativo:
# user_input = input("Usuário, digite a sua pergunta: ")
# print(llm(user_input))


"""
* Com T5-Flan-XL
"""
from langchain.llms import HuggingFaceHub

llm_hf = HuggingFaceHub(
    repo_id="google/flan-t5-xl",
    model_kwargs={"temperature":0.0}
                       )


# Fazemos interativo:
user_input = input("Usuário, digite a sua pergunta: ")
print(llm_hf(user_input))



"""
Prompt Templates
----------------
"""

from langchain import PromptTemplate

restaurant_template = """
Quero que você atue como consultor de nomes para novos restaurantes.

Retorna uma lista de nomes de restaurantes. Cada nome deve ser curto,
cativante e fácil de lembrar. Deve estar relacionado ao tipo de restaurante que você está nomeando.

Quais são alguns bons nomes para um restaurante que é {restaurant_description}?
"""

prompt_template = PromptTemplate(
    input_variables=["restaurant_description"],
    template=restaurant_template
                       )

# Alguns exemplos de descrição de restaurantes:
description = "um lugar grego que serve souvlakis de cordeiro fresco e outras comidas gregas."
description_02 = "uma lanchonete com tema de memorabilia de beisebol."
description_03 = "um café que tem música hard rock ao vivo e lembranças."

# Vejamos como é a formatação de nosso Prompt:
user_input = input("Digite a descrição do seu restaurante: ")
print(prompt_template.format(restaurant_description=user_input))


# Vamos fazer a QUERY ao Modelo e com o Prompt Template que criamos:
from langchain.chains import LLMChain

chain = LLMChain(llm=llm,
                 prompt=prompt_template
                )

# Executamos a nossa CADEIA especificando apenas a variável de entrada:
print(chain.run(user_input))

"""
Few-Shot Learning
"""
from langchain import PromptTemplate, FewShotPromptTemplate

# PRIMEIRO, criamos a lista de alguns exemplos de Few-Shot:
examples = [
    {"word": "Feliz", "antonym": "Triste"},
    {"word": "Alto", "antonym": "Baixo"},
           ]

# EM SEGUIDA, especificamos o template para formatar os exemplos que fornecemos.
example_formatter_template = """
                                Palavra: {word}
                                Antônimo: {antonym}\n
                            """

# Usamos a classe `PromptTemplate` para isso.
example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template,
                               )

# Finalmente, criamos o objeto `FewShotPromptTemplate`:
few_shot_prompt = FewShotPromptTemplate(
    # Os exemplos que queremos inserir no prompt:
    examples=examples,
    # É assim como são formatados os exemplos quando os inserimos no Prompt:
    example_prompt=example_prompt,
    # O PREFIXO é algum texto que vem ANTES dos exemplos no promp.
    # Geralmente, isto consiste em Instruções:
    prefix="Dê o antônimo de cada entrada",
    # O SUFIXO é algum texto que vai DEPOIS dos exemplos no prompt.
    # Geralmente, é aqui que a entrada do Usuário irá: 
    suffix="Palavra: {input}\nAntônimo:",
    # As variáveis de entrada são as variáveis que o prompt geral espera.
    input_variables=["input"],
    # O `example_separator` é a string que usaremos para unir o PREFIXO, os EXAMPLES e o SUFIXO.
    example_separator="\n\n",
)

# Visualizamos a formatação de nosso Prompt (para isso usamos o método 'format'):
user_input = input("Digite a palavra da qual quer seu antônimo: ")
print(few_shot_prompt.format(input=user_input))


from langchain.chains import LLMChain

chain = LLMChain(llm=llm,
                 prompt=few_shot_prompt
                )

# Executamos a cadeia especificando apenas a variável de entrada:
print(chain.run(user_input))
