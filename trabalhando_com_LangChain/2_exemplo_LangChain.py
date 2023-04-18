"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro
"""

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain


# Isto é quando usas o arquivo .env:
from dotenv import load_dotenv
import os
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 



from langchain import FewShotPromptTemplate

openai_llm = OpenAI(temperature=0.0,
                    verbose=True)  

examples = [
            {"name":"Spotify", "category":"música"},
            {"name":"Instagram", "category":"social"},
            {"name":"TikTok", "category":"social"},
            {"name":"Tinder", "category":"namoro"},
            {"name":"TalkFamous", "category":"entretenimento"}
           ]

app_prompt = PromptTemplate(
    input_variables=["name", "category"],
    template="""
    Categoria: {category}
    Nome do aplicativo: {name}
    """
                           )


# Agora crie o Modelo de Prompt de Poucos Tiros
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=app_prompt,
    prefix="Crie nomes para aplicativos com base em sua categoria.",
    suffix="Categoria: {category}\nNome do aplicativo.",
    input_variables=["category"],
    example_separator="\n\n"
                                                )


# prompt_formatado = few_shot_prompt_template.format(category="estilo de vida")
# print(prompt_formatado)



chain = LLMChain(llm=openai_llm,
                 prompt=few_shot_prompt_template
                )


user_input = input("Digite a sua CATEGORIA: ")
print(chain.run(user_input))
