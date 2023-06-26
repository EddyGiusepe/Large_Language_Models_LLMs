"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Iisdro

Objetivo: Aqui reproduzimos o mesmo script do Vishnu Sivan para
          criar um aplicativo de resumo de PDFs usando LangChain e Gradio 🤗.

Executar este script:
                     $ python main2.py
"""
import gradio as gr
from langchain import OpenAI, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

import os
import openai
from dotenv import load_dotenv
load_dotenv()
Eddy_API_KEY_OpenAI = os.environ['OPENAI_API_KEY'] 


llm = OpenAI(model_name='text-davinci-003', # text-davinci-003
             temperature=0,
             max_tokens=150
            )

def summarize_pdf(path):
    """Esta função aceita um PATH de arquivo PDF para gerar um RESUMO conciso."""
    summary = ""
    try:
        loader = PyPDFLoader(path.name) # Carrega o conteúdo PDF
        docs = loader.load_and_split()
        #chain = load_summarize_chain(llm, chain_type="map_reduce") # Cria uma cadeia de resumo
        #summary = chain.run(docs)
        #print(summary)
        prompt_template = """
        Nesta tarefa de resumo, você deve usar SÓ o idioma Português do Brasil \
        para fornecer o RESUMO do ```{text}```. Depois do resumo adicione um \
        emoji de cara feliz.
        """

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

        chain = load_summarize_chain(llm,
                                     chain_type="map_reduce",
                                     map_prompt=PROMPT,
                                     combine_prompt=PROMPT,
                                     verbose=False
                                    )
        summary = chain.run(docs)
    except:
        summary = "Algo deu errado. \nPor favor, tente com algum outro documento."
    return summary
    
def upload_file(file):
    return file.name

def main():
    global input_pdf_path
    with gr.Blocks() as demo:
        file_output = gr.File()
        upload_button = gr.UploadButton("Clique para fazer Upload de um arquivo", file_types=["pdf"])
        upload_button.upload(upload_file, upload_button, file_output)

    output_summary = gr.Textbox(label="Resumo")

    interface = gr.Interface(
        fn=summarize_pdf,
        inputs=[upload_button],
        outputs=[output_summary],
        title="Resumidor de PDF",
        description="",
    )
    
    interface.launch()

if __name__ == "__main__":
    main()
