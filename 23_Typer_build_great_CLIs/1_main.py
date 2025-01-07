#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Typer
=====
Typer é uma biblioteca poderosa e intuitiva para a 
construção de aplicações de linha de comando (CLI) 
em Python, projetada para ser fácil de usar tanto 
para desenvolvedores quanto para usuários finais.
Baseada em type hints do Python, Typer permite que 
você crie CLIs com um mínimo de código e complexidade.

NOTA:
=====
* Use typer.run() quando tiver apenas uma função

* Use app = typer.Typer() quando precisar de múltiplos comandos

* Quando você está usando o typer.Typer(), você precisa
usar o decorador @app.command() para cada função que 
você quer expor como um comando CLI. O decorador @app.command() 
é necessário porque:
  1. Ele marca a função como um subcomando da sua CLI
  2. Permite que o Typer processe os argumentos e opções automaticamente
  3. Adiciona a função à documentação de ajuda (--help)

Exemplos de uso
---------------
# Para ver a ajuda:
python main.py --help

# Para somar dois números (por exemplo, 5 e 3):
python main.py 5 3
"""
import typer

app = typer.Typer()

@app.command()
def somar(
    numero1: int = typer.Argument(..., help="Primeiro número inteiro"),
    numero2: int = typer.Argument(..., help="Segundo número inteiro")
):
    """Soma dois números e mostra o resultado"""
    resultado = numero1 + numero2
    typer.echo(f"A soma de {numero1} + {numero2} = {resultado}")
    #return f"A soma de {numero1} + {numero2} = {resultado}"

if __name__ == "__main__":
    app()