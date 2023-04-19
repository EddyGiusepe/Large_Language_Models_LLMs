"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro

A ideia:
        Aqui vamos aplicar Encadeamento de PROMPTS para resolver um problema complexo.
        Sabemos que cada chamada de prompt tem uma entrada e uma saída. Você pode alimentar
        a saída de um prompt como entrada para outro prompt.   
Nota:
     Baseado no artigo de Zhao Guodong --> https://bootcamp.uxdesign.cc/advanced-skills-in-gpt-for-your-product-business-beyond-simple-chats-b6cca061187e             
"""
from langchain.chains import SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain import FewShotPromptTemplate


import os
os.environ["OPENAI_API_KEY"] = "CHAVE_API_KEY"

llm = OpenAI(temperature=0.0,
             verbose=True
            )


article = """
A Coinbase, a segunda maior bolsa de criptomoedas em volume de negócios, divulgou seus resultados do quarto
trimestre de 2022 na terça-feira, dando aos acionistas e participantes do mercado uma visão atualizada de suas
finanças. Em resposta ao relatório, as ações da empresa caíram modestamente no início do pregão. No quarto trimestre
de 2022, a Coinbase gerou $ 605 milhões em receita total, bem abaixo dos $ 2,49 bilhões no trimestre do ano anterior.
A linha principal da Coinbase não foi suficiente para cobrir suas despesas: a empresa perdeu $ 557 milhões no período
de três meses em uma base GAAP (lucro líquido) no valor de - $ 2,46 por ação e um déficit de EBITDA ajustado de
$124 milhões. Wall Street esperava que a Coinbase parasse relatam $ 581,2 milhões em receita e ganhos por
ação de - $2,44 com EBITDA ajustado de - $ 201,8 milhões impulsionados por 8,4 milhões de usuários de transações
mensais (MTUs), de acordo com dados fornecidos pelo Yahoo Finance. Antes de seus ganhos do quarto trimestre serem
divulgados, as ações da Coinbase haviam subido 86 % no acumulado do ano. Mesmo com esse rali, o valor da Coinbase
quando medido por ação ainda está significativamente abaixo de sua alta de 52 semanas de $ 206,79. Que a Coinbase
superou as expectativas de receita é notável porque veio com quedas no volume de negociação; A Coinbase historicamente
gerou a maior parte de suas receitas com taxas de negociação, tornando o quarto trimestre de 2022 notável. 
"""
# Definimos o nosso P Prompt:
first_prompt = PromptTemplate(
    input_variables=["query"],
    template="Estes Dados são informações das quais os usuários podem fazer perguntas respeito a eles. Responda às perguntas que o usuário fazer de maneira concisa e clara. Não inclua opinióes. :\n\n {query}",
                             )
    # template="Extraia os fatos principais deste texto. Não inclua opiniões. Dê a cada fato um número e mantenha-os em frases curtas. :\n\n {query}"

# Definimos a nossa PRIMEIRA cadeia:
chain_one = LLMChain(llm=llm,
                     prompt=first_prompt
                    )
result_1 = chain_one.run(article)



# Criamos nossos exemplos para realizar um Few-Shot Learning:
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
# Criar um exemplo de Template --> second_prompt:
example_template = """
User: {query}
AI: {answer}
"""
second_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
                              )
# Agora dividimos nosso Prompt anterior em um Prefixo e um Sufixo
# O prefixo é nossas instruções
prefix = """Responda a pergunta com base no contexto abaixo e com base ao contexto do bate papo entre o usuário e o assistente.
            Se a pergunta não puder ser respondida usando as informações fornecidas, responda com "Não sei".
            Aqui estão alguns exemplos: 
         """

# E o sufixo é nosso indicador de entrada e saída do usuário
suffix = """
User: {query}
AI: """

# Agora crie o Modelo de Prompt de Poucos Tiros
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=second_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

# Printamos a FORMATAÇÃO de nosso Prompt:
print(few_shot_prompt_template.format(query="")) 

# Definimos a nossa SEGUNDA cadeia:
chain_two = LLMChain(llm=llm,
                     prompt=few_shot_prompt_template
                    )

result_2 = chain_two.run(result_1)


# Jutamos as nossas duas cadeias:
full_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                     verbose=True
                                  )


# Executamo a CADEIA especificando apenas a VARIÁVEL DE ENTRADA para a primeira cadeia:
# user_input = input("Usuário, digite a sua pergunta: ")
# print(full_chain.run(user_input))
print("Digite a sua pergunta para começar uma conversa com a AI: ")
while True:
    user_input = input()
    result = full_chain.run(user_input)
    print(result)

    if not user_input:
        break