"""
Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro

Este Aplicativo foi baseado na publicação de 🤗"Christophe Atten"🤗, publicado no 📑 DataDrivenInvestor 📑. 

Modo de execução: 
                 $ streamlit run nome_do_script.py
"""
import streamlit as st
import openai
import PyPDF2
import io
import re
import os
from dotenv import find_dotenv, load_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.getenv('OPENAI_API_KEY')

def get_completion(prompt, model="gpt-3.5-turbo-16k"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
     model=model,
     messages=messages,
     temperature=0, 
    )
    return response.choices[0].message["content"]

def your_summarizer_function(text):
    iterations = len(text)+1
    result = []
    summary=' '
    for i in range(len(text)):
        prompt =f"""
        Resuma o seguinte texto: \
        Extraia informações relevantes do texto a seguir, delimitado por crases triplos.\
        Certifique-se de preservar os detalhes importantes.
        Texto: ```{text[i]}```
        """
        try:
            response = get_completion(prompt)
        except:
            response = get_completion(prompt)
        summary= summary+' ' +response +'\n\n'
        result.append(response)
        progress = (i+1) / iterations
        progress_bar.progress(progress)

    prompt =f"""
    Resuma o seguinte texto: \
    Extraia informações relevantes do texto a seguir, delimitado por crases triplos.\
    Certifique-se de preservar os detalhes importantes.
    Texto: ```{result}```
    """
    try:
        response = get_completion(prompt)
    except:
        response = get_completion(prompt)
    progress = (i+2) / iterations
    progress_bar.progress(progress)
    return response

def pdf_to_text(file):
    # Criando um objeto de leitura de PDF:
    pdfReader = PyPDF2.PdfReader(file)
    text=[]
    # Armazenando as páginas em uma lista:
    for i in range(0,len(pdfReader.pages)):
      # Criando um objeto de página:
      pageObj = pdfReader.pages[i].extract_text()
      pageObj= pageObj.replace('\t\r','')
      pageObj= pageObj.replace('\xa0','')
      # Extraindo texto da página
      text.append(pageObj)
    return text


st.title("Resumidor de PDF")

uploaded_file = st.file_uploader("Escolha um arquivo de PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner('Lendo PDF ...'):
        pdf_text = pdf_to_text(uploaded_file)
        st.success('PDF lido com sucesso!')
        num_words = sum(len(re.findall(r'\b\w+\b', page)) for page in pdf_text)  # Contar palavras em cada página
        st.write(f'Número de páginas: {len(pdf_text)}')
        st.write(f'Número de palavras: {num_words}')
    
    if st.button('Resumir'):
        progress_bar = st.progress(0)
        with st.spinner('Resumindo...'):
            final_summary = your_summarizer_function(pdf_text)
            st.success('Resumo gerado com sucesso!')
            st.write(final_summary)
else:
    st.write("Faça o Upload de um arquivo PDF")
