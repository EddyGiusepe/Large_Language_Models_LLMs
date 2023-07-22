"""
Método de execução:
===================
Primeiro você deve instalar alguns pacotes necessários.
Só precisamos de openai e langchain 🤗.

$ pip install gradio --upgrade
$ python app.py
"""
import gradio as gr
import os
import openai
import gradio as gr
from gradio import ChatInterface
import time

from dotenv import find_dotenv, load_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.getenv('OPENAI_API_KEY')

# Import things that are needed generically from langchain
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.tools import MoveFileTool, format_tool_to_openai_function
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import AIPluginTool

def predict(inputs, chatbot):

    print(chatbot)

    messages = []
    for conv in chatbot:
        user = conv[0]
        messages.append({"role": "user", "content":user })
        assistant = conv[1]
        messages.append({"role": "assistant", "content":assistant})
    messages.append({"role": "user", "content": inputs})

    # a ChatCompletion request
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages= messages, # example :  [{'role': 'user', 'content': "What is life? Answer in three words."}],
        temperature=0.0,
        stream=True  # for streaming the output to chatbot
    )

    partial_message = ""
    for chunk in response:
        if len(chunk['choices'][0]['delta']) != 0:
          print(chunk['choices'][0]['delta']['content'])
          partial_message = partial_message + chunk['choices'][0]['delta']['content']
          yield partial_message 

gr.ChatInterface(predict).queue().launch()
