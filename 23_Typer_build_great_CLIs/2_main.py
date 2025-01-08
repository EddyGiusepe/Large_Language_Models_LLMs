#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Aplicação CLI, usando Typer, para manipulação de texto
======================================================
Esta é uma aplicação CLI (Interface de Linha de Comando)
desenvolvida com o framework Typer em Python, que oferece
três comandos principais: converter para transformar textos
em maiúsculas ou minúsculas, exibir para mostrar mensagens
coloridas no terminal, e info para apresentar informações 
sobre a aplicação. A interface é intuitiva e inclui mensagens
de ajuda detalhadas para cada comando, permitindo que usuários
manipulem textos de forma simples através do terminal.


Comando converter
=================
# Formato básico
python 2_main.py converter "seu texto aqui"

# Exemplos:
python 2_main.py converter "Olá Mundo"  # converte para maiúsculas (padrão)
python 2_main.py converter "Olá Mundo" --modo minusculas  # converte para minúsculas

Para ver ajuda específica de cada comando, você pode usar:
* python 2_main.py converter --help
* python 2_main.py exibir --help
* python 2_main.py info --help

Comando exibir
==============
# Formato básico
python 2_main.py exibir "sua mensagem" --cor "nome_da_cor"

# Exemplo:
python 2_main.py exibir "Olá Mundo" --cor blue
python 2_main.py exibir "Atenção!" --cor red

Comando info
============
# Mostra informações sobre a aplicação
python 2_main.py info
"""
import typer

# Criamos uma instância do Typer com opções adicionais:
app = typer.Typer(
    name="MeuApp",
    help="Uma aplicação CLI para manipulação de texto.",
    add_completion=False  # Habilita a funcionalidade de autocompletar comandos no terminal.
                         # Isso significa que você pode usar a tecla TAB para completar automaticamente comandos, opções e argumentos da sua CLI.
)

# Comando para exibir informações sobre a aplicação:
@app.command()
def info():
    """Mostra informações sobre a aplicação."""
    typer.secho("Esta é uma aplicação CLI simples feita com Typer.", fg=typer.colors.BLUE)
    typer.echo("Você pode usar os comandos 'exibir' e 'converter'.")

# Comando para converter texto:
@app.command()
def converter(
    texto: str = typer.Argument(..., help="O texto a ser convertido."),
    modo: str = typer.Option(default="maiusculas", help="Modo de conversão: 'maiusculas' ou 'minusculas'.")
):
    """Converte o texto para maiúsculas ou minúsculas (default: maiusculas)."""
    if modo == "maiusculas":
        resultado = texto.upper()
        typer.secho(f"Texto convertido: {resultado}", fg=typer.colors.GREEN)
    elif modo == "minusculas":
        resultado = texto.lower()
        typer.secho(f"Texto convertido: {resultado}", fg=typer.colors.GREEN)
    else:
        raise typer.BadParameter("Modo deve ser 'maiusculas' ou 'minusculas'.")
    
# Comando para exibir mensagem em uma cor específica:
@app.command()
def exibir(
    mensagem: str = typer.Argument(..., help="A mensagem a ser exibida."),
    cor: str = typer.Option(default="red", help="A cor da mensagem (red, green, blue, etc.).")
):
    """Exibe uma mensagem na cor especificada."""
    try:
        # Exibe a mensagem na cor escolhida
        typer.secho(mensagem, fg=cor)
    except Exception as e:
        # Em caso de erro, exibe uma mensagem de erro
        typer.echo(f"Erro ao exibir a mensagem: {e}", fg=typer.colors.RED)




if __name__ == "__main__":
    app()
