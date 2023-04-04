"""
Este é um script Python que serve como interface para um modelo de IA conversacional construído com as bibliotecas `LangChain` e `LLMs`.
O código cria um aplicativo da Web usando Streamlit, uma biblioteca Python para criar aplicativos da Web interativos.
# Author: Avratanu Biswas
# Date: March 11, 2023

Link de estudo: https://medium.com/@avra42/how-to-build-a-chatbot-with-chatgpt-api-and-a-conversational-memory-in-python-8d856cda4542
"""

# Import necessary libraries
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Set Streamlit page configuration
st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')
# Inicializar estados de sessão
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# Definir função para obter a entrada do usuário
def get_text():
    """
    Obtenha o texto de entrada do Usuário.
    Returns:
        (str): O texto digitado pelo Usuário
    """
    input_text = st.text_input("Você: ", st.session_state["input"], key="input",
                            placeholder="Seu assistente de AI aqui! pergunte-me qualquer coisa ...", 
                            label_visibility='hidden')
    return input_text

# Define function to start a new chat
def new_chat():
    """
    Limpa o estado da sessão e inicia um novo chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()

# Configure a barra lateral com várias opções
with st.sidebar.expander("🛠️ ", expanded=False):
    # Opção para visualizar o armazenamento de memória
    if st.checkbox("Visualizar armazenamento de memória"):
        with st.expander("Memory-Store", expanded=False):
            st.session_state.entity_memory.store
    # Opção para visualizar o buffer de memória
    if st.checkbox("Visualizar buffer de memória"):
        with st.expander("Bufffer-Store", expanded=False):
            st.session_state.entity_memory.buffer
    MODEL = st.selectbox(label='Modelo', options=['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])
    K = st.number_input(' (#)Resumo dos prompts a serem considerados',min_value=3,max_value=1000)

# Configurar o layout do aplicativo Streamlit
st.title("🤖 Chat Bot com 🧠")
st.subheader(" Alimentado pela 🦜 LangChain + OpenAI + Streamlit")

# Ask the user to enter their OpenAI API key
API_O = st.sidebar.text_input("API-KEY", type="password")

# Session state storage would be ideal
if API_O:
    # Create an OpenAI instance
    llm = ChatOpenAI(temperature=0,
                openai_api_key=API_O, 
                model_name=MODEL, 
                verbose=False) 


    # Crie um objeto ConversationEntityMemory se ainda não tiver sido criado
    if 'entity_memory' not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K )
        
        # Create the ConversationChain object with the specified configuration
    Conversation = ConversationChain(
            llm=llm, 
            prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
            memory=st.session_state.entity_memory
        )  
else:
    st.sidebar.warning('Chave de API necessária para experimentar este aplicativo. A chave de API não está armazenada de nenhuma forma.')
    # st.stop()


# Add a button to start a new chat
st.sidebar.button("New Chat", on_click = new_chat, type='primary')

# Get the user input
user_input = get_text()

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:
    output = Conversation.run(input=user_input)  
    st.session_state.past.append(user_input)  
    st.session_state.generated.append(output)  

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    
    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download',download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session