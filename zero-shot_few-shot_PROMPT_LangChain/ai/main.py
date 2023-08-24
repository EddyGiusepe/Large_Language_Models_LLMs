"""
Script baseado no tutorial de "Sourav Bhattacharjee"

Data Scientist.: Dr.Eddy Giusepe Chirinos Isidro 
"""
from completion.utility import utility
from prompts.summarize import summarize

# Executamos o script: $ python main.py
if __name__ == '__main__':

    #summarizer = summarize.Summarizer(prompt_type="G") # Exemplo 1
    #summarizer = summarize.Summarizer(prompt_type="S") # Exemplo 2
    summarizer = summarize.Summarizer(prompt_type="EddyGiusepe") # Exemplo 3
    text = summarize.Summarizer.get_text()
    prompt = summarizer.get_prompt(text)

    # Exemplo 1:
    # genericPrompt = utility.AIUtility(prompt=prompt)
    # genericPrompt.print_completion()

    # Exemplo 2:
    #specificPrompt = utility.AIUtility(prompt=prompt)
    #specificPrompt.print_completion()

    # Exemplo 3:
    EddyPrompt = utility.AIUtility(prompt=prompt)
    EddyPrompt.print_completion()
    