"""
Script baseado no tutorial de "Sourav Bhattacharjee"

Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro 
"""
import os

cwd = os.getcwd()
relativePath = cwd + '/prompts/summarize/'


class Summarizer:
    def __init__(self, prompt_type):
        self.prompt_type = prompt_type

    def __str__(self):
        return f"{self.prompt_type}"

    @staticmethod
    def get_text():
        with open(relativePath + 'text.txt') as f:
            contents = f.read()
            return contents

    def get_prompt(self, text):
        match self.prompt_type:
            case "G":
                with open(relativePath + 'genericprompt.txt') as f:
                    contents = f.read()
                    return contents + text
            case "S":
                with open(relativePath + 'specificprompt.txt') as f:
                    contents = f.read()
                    return contents + text
            case "EddyGiusepe":
                with open(relativePath + 'Eddyprompt.txt') as f:
                    contents = f.read()
                    return contents + text
            case default:
                return "not a valid input"
            