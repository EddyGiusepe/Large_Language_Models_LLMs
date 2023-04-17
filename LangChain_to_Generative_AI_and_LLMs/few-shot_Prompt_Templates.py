"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro


A ideia:
        usamos "FewShotPromptTemplate" para fornecer treinamento de Few-Shot como
        "conhecimento da fonte". Para fazer isso, adicionamos alguns exemplos a
        nosso prompt que o modelo pode ler e aplicar à entrada do usuário.

Caso de Uso:
            configuramos alguns exemplos de Few-Shot para autopergunta (self-ask) com pesquisa.           
"""

# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
print('Carregando a minha chave Key: ', load_dotenv())
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]

from langchain.llms import OpenAI


# Instanciamos o Modelo 
openai = OpenAI(
    model_name="text-davinci-003",
    openai_api_key=Eddy_API_KEY_OpenAI
)


from langchain import FewShotPromptTemplate
from langchain import PromptTemplate

# Criamos nossos exemplos
examples = [
    {
        "query": "Bom dia AI?",
        "answer": "Bom dia! Como posso ajudar?"
    }, 
    {
        "query": "Qual é a idade mínima para crianças frequentarem uma creche aqui em Brasília DF?",
        "answer": "De 0 a 3 anos de idade. Os pediatras, também, recomendam a partir de 2 anos. "
    },
    {
        "query": "Qual é a carga horária nas creches de Brasília DF?",
        "answer": "O atendimento mínimo é de 4 horas ao dia e o período integral é de 7 horas ao dia."
    },
    {
        "query": "É preciso levar comida para creche?",
        "answer": "Você precisa perguntar se sua creche oferece refeições ou se você precisará tazer comida diariamente."
    },
    {
        "query": "Quais são as medidas de segurança nas creches de Brasília DF?",
        "answer": "As medidas de segurança para crianças frequentarem uma creche em Brasília DF incluem a verificação dos pais, verificação de saúde das crianças, a verificaçãos de vacinas, a verificação de segurança do local, etc."
    }
           ]

# Criar um exemplo de Template
example_template = """
User: {query}
AI: {answer}
"""

# Crie um exemplo de prompt a partir do Template de acima
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# Agora dividimos nosso Prompt anterior em um Prefixo e um Sufixo
# O prefixo é nossas instruções
prefix = """Responda a pergunta com base no contexto abaixo. Se a pergunta não puder ser respondida
            usando as informações fornecidas, responda com "Não sei". Aqui estão alguns exemplos: 
         """

# E o sufixo é nosso indicador de entrada e saída do usuário
suffix = """
User: {query}
AI: """

# Agora crie o Modelo de Prompt de Poucos Tiros
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)



print("Digite a sua pergunta para começar uma conversa com a AI: ")
while True:
    input_question = input()
    
    print(openai(few_shot_prompt_template.format(query= input_question)
          ))
    if not input_question:
        break
     