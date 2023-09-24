"""
Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro

Executar assim: $ streamlit run openai_streaming

OBS: O modelo que estamos usando aqui é antigo (DEPRECATED), tenta usar 
     um modelo mais recente 🧐.
"""
import openai 
import streamlit as st  

# YOU MUST HAVE USER ENV VARAIABLE SET FOR "OPENAI_API_KEY"
import os
from dotenv import find_dotenv, load_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.getenv('OPENAI_API_KEY')

st.title('🤗 Obtendo respostas em Streaming 🤗')
with st.sidebar:
    # Escrevendo um título na página:

    # Nome do autor:
    st.write("Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro")
    "[e-mail: eddychirinos.unac@gmail.com]()"
placeholder_response_user_input = st.empty()
user_input = placeholder_response_user_input.text_input("Digite sua pergunta aqui: ", key= "user_input")



print("App iniciado")
# streaming_response = []
completion_text = ''

placeholder_response = st.empty()


# put a button to submit the text input
if user_input:

    placeholder_response.text("Aguardando resposta...")
    print('Obtendo resposta')
    response = openai.Completion.create(
    model='text-davinci-003',
    prompt=user_input,
    max_tokens=50,
    temperature=0,
    stream=True,  # HERE WE SET STREAMING TO TRUE
)
    
    for r in response:
        r_text = r['choices'][0]['text']  
        completion_text += r_text  

        # write the text to the web app
        placeholder_response.markdown(completion_text)

    print("end")


# print(f"Full text received: {completion_text}")
