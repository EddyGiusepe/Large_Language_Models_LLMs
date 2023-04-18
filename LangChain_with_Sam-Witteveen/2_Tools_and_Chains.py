"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro
"""
# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 
Eddy_API_KEY_HuggingFace = os.environ["HUGGINGFACEHUB_API_TOKEN"]


"""
Tools e Chains com LnagChain
----------------------------

* LLMChain básico para Extração de fatos
"""
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(model_name='text-davinci-003', 
             temperature=0, 
             max_tokens = 256)

# Vamos pegar uma pequena parte de um texto (para não consumir muitos Tokens):
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
# Os volumes
# de negociação do consumidor caíram de US$ 26 bilhões no terceiro trimestre do ano passado para US$ 20 bilhões no quarto
# trimestre, enquanto os volumes institucionais no mesmo período caíram de US$ 133 bilhões para US$ 125 bilhões,
# o que resultou em volumes totais de negociação e receitas de transações da Coinbase caindo 50% e 66% ano a ano,
# respectivamente, informou a empresa. terceiro trimestre do ano passado, caindo de US$ 365,9 milhões para US$322,1 milhões.
# (O TechCrunch está comparando os resultados do quarto trimestre de 2022 da Coinbase com o terceiro trimestre de 2022,
# em vez do quarto trimestre de 2021, pois a última comparação seria menos útil, considerando o quanto o mercado de
# criptografia mudou no ano passado; todos sabemos que a atividade geral de criptografia caiu do últimos meses de 2021.)
# Houve boas notícias no relatório da Coinbase. Embora as receitas comerciais da Coinbase tenham sido menos do que
# exuberantes, as outras receitas da empresa registraram ganhos. O que a Coinbase chama de "receita de assinaturas e
# serviços" aumentou de US$ 210,5 milhões no terceiro trimestre de 2022 para US$ 282,8 milhões no quarto trimestre
# do mesmo ano, um ganho de pouco mais de 34% em um único trimestre. eventos, incluindo os colapsos de Terra/LUNA e
# FTX para citar alguns, ainda houve crescimento em outras áreas. Os desenvolvedores ativos mensais em cripto mais do
# que dobraram desde 2020 para mais de 20.000, enquanto grandes marcas como Starbucks, Nike e Adidas mergulharam no
# espaço ao lado de plataformas de mídia social como Instagram e Reddit. esse movimento resulta em maior adoção tanto
# para casos de uso de produtos quanto para volumes de negociação. Embora tenha havido muito movimento nos mercados
# de varejo tradicionais e nos negócios da Web 2.0, o volume de negócios para consumidores e usuários institucionais
# caiu trimestre a trimestre para a Coinbase. o interesse comercial ressurgir em 2023, ou se plataformas como a Coinbase
# tiverem que continuar procurando receita em outro lugar (como seu serviço de assinatura) se os usuários continuarem
# a se afastar do mercado.

print(len(article))

fact_extraction_prompt = PromptTemplate(
    input_variables=["text_input"],
    template="Extraia os fatos principais deste texto. Não inclua opiniões. Dê a cada fato um número e mantenha-os em frases curtas. :\n\n {text_input}"
)

# A nossa PRIMEIRA cadeia:
fact_extraction_chain = LLMChain(llm=llm,
                                 prompt=fact_extraction_prompt
                                )

facts = fact_extraction_chain.run(article)

print(facts)


"""
* Reescrevemos como um RESUMO de fatos
"""
investor_update_prompt = PromptTemplate(
    input_variables=["facts"],
    template="Você é um analista do Goldman Sachs. Pegue a lista de fatos a seguir e use-a para escrever um parágrafo curto para os investidores. Não deixe de fora informações importantes:\n\n {facts}"
)

# Aqui temos a nossa SEGUNDA cadeia:
investor_update_chain = LLMChain(llm=llm,
                                 prompt=investor_update_prompt
                                )

investor_update = investor_update_chain.run(facts)

print(investor_update)
print("")
len(investor_update)


"""
* Criando trios para plots
"""
triples_prompt = PromptTemplate(
    input_variables=["facts"],
    template="Pegue a seguinte lista de fatos e transforme-os em trios para um gráfico de conhecimento (knowledge Graph):\n\n {facts}"
                               )

# Aqui temos a nossa TERCEIRA cadeia:
triples_chain = LLMChain(llm=llm,
                         prompt=triples_prompt
                        )

triples = triples_chain.run(facts)

print(triples)
print("")
len(triples)


"""
Encadeando as chains anteriores
-------------------------------
"""
from langchain.chains import SimpleSequentialChain, SequentialChain

full_chain = SimpleSequentialChain(chains=[fact_extraction_chain, investor_update_chain],
                                   verbose=True
                                  )

response = full_chain.run(article)
print(response)

