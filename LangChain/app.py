import streamlit as st 
from langchain.chains import LLMChain, SimpleSequentialChain # Importando bibliotecas LangChain
from langchain.llms import OpenAI # Importando o Modelo OpenAI
from langchain.prompts import PromptTemplate # Importando PromptTemplate
import openai

# título da nossa APP
st.title("✅ O que é verdade  : Usando LangChain `SimpleSequentialChain`")

# Adionamos o Link na qual foi baseado este desenho
st.markdown("Inspirado em [fact-checker](https://github.com/jagilley/fact-checker) por Jagiley")


# Isto é quando usas o arquivo .env: 
from dotenv import load_dotenv
import os
print('Carregando a minha chave Key: ', load_dotenv())
Eddy_API_KEY = os.environ['OPENAI_API_KEY']  
openai.api_key = Eddy_API_KEY 


# Se uma chave de API foi fornecida, crie uma instância de modelo de linguagem OpenAI
if Eddy_API_KEY:
    llm = OpenAI(temperature=0.7,
                 openai_api_key=Eddy_API_KEY)
else:
    # Se uma chave de API não foi fornecida, exibe uma mensagem de aviso
    st.warning("Enter your OPENAI API-KEY. Get your OpenAI API key from [here](https://platform.openai.com/account/api-keys).\n")


# Adicionamos uma caixa de entrada de texto para a pergunta do usuário
user_question = st.text_input(
    "Digite a sua pergunta : ",
    placeholder = "Nossa maior fraqueza está em desistir. O caminho mais certo de vencer é tentar mais uma vez.",
)


# Gerando a resposta final para a pergunta do usuário usando todas as cadeias
if st.button("Conte-me sobre isso", type="primary"):
    # Chain 1: Gerando uma versão reformulada da pergunta do usuário
    template = """{question}\n\n"""
    prompt_template = PromptTemplate(input_variables=["question"], template=template)
    question_chain = LLMChain(llm=llm, prompt=prompt_template)
    #st.success(question_chain.run(user_question)) # <-- Eddy Add

    # Chain 2: Gerando suposições feitas na declaração
    template = """Aqui está uma declaração:
        {statement}
        Faça uma lista com marcadores das suposições que você fez ao produzir a declaração acima.\n\n"""
    prompt_template = PromptTemplate(input_variables=["statement"], template=template)
    assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)
    assumptions_chain_seq = SimpleSequentialChain(
        chains=[question_chain, assumptions_chain], verbose=True
    )
    #st.success(assumptions_chain_seq.run(user_question)) # <-- Eddy Add

    # Chain 3: Verificando os fatos as suposições
    template = """Aqui está uma lista de pontos de afirmações:
    {assertions}
    Para cada afirmação, determine se é verdadeira ou falsa. Se for falso, explique por que.\n\n"""
    prompt_template = PromptTemplate(input_variables=["assertions"], template=template)
    fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)
    fact_checker_chain_seq = SimpleSequentialChain(
        chains=[question_chain, assumptions_chain, fact_checker_chain], verbose=True
    )

    # Final Chain: Gerar a resposta final para a pergunta do usuário com base nos fatos e suposições
    template = """À luz dos fatos acima, como você responderia à pergunta '{}'""".format(
        user_question
    )
    template = """{facts}\n""" + template
    prompt_template = PromptTemplate(input_variables=["facts"], template=template)
    answer_chain = LLMChain(llm=llm, prompt=prompt_template)
    overall_chain = SimpleSequentialChain(
        chains=[question_chain, assumptions_chain, fact_checker_chain, answer_chain],
        verbose=True,
    )

    # Executando todas as cadeias na pergunta do usuário e exibindo a resposta final
    st.success(overall_chain.run(user_question))

