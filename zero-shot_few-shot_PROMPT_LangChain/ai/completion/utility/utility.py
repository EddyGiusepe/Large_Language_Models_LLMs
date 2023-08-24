"""
Script baseado no tutorial de "Sourav Bhattacharjee"

Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro 
"""
import os
import openai
from dotenv import load_dotenv, find_dotenv

class AIUtility:
    _ = load_dotenv(find_dotenv()) # read local .env file
    openai.api_key  = os.environ['OPENAI_API_KEY']

    def __init__(self, prompt, temperature=0, model="gpt-3.5-turbo"):
        self.prompt = prompt
        self.temperature = temperature
        self.model = model

    def __str__(self):
        return f"{self.prompt}"

    def get_completion(self):
        messages = [{"role": "user", "content": self.prompt}]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        return response.choices[0].message["content"]

    def print_completion(self):
        # Use um breakpoint na linha de código abaixo para depurar seu script.
        response = self.get_completion()  # Press ⌘F8 to toggle the breakpoint.
        print(response)
        print("-----------------------------------------------------------------")
