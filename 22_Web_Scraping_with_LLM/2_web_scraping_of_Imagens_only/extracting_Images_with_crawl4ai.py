#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Web Scraping Assíncrono com Crawl4AI
====================================
Este script é um web scraper que acessa especificamente a página de cursos de tecnologia
da KidoCode (https://www.kidocode.com/degrees/technology), extrai informações como texto,
links e imagens da página. O script utiliza a biblioteca crawl4ai para fazer o scraping 
de forma assíncrona (mais eficiente) e exclui elementos desnecessários como navegação e 
rodapé durante a extração.Por fim, todos os dados coletados são salvos em um arquivo JSON 
chamado 'resultados_cursos.json' para posterior análise ou uso.
"""
from dataclasses import dataclass
from typing import List
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

@dataclass
class Curso:
    """
    Estrutura de dados para armazenar informações de um curso.
    
    Attributes:
        titulo (str): Nome do curso
        descricao (str): Descrição detalhada do curso
        url (str): URL da página do curso
        imagem (str): URL da imagem do curso
    """
    titulo: str
    descricao: str
    url: str
    imagem: str

class AnalisadorCursos:
    """
    Classe responsável por analisar e extrair informações de cursos da página web.
    """
    def __init__(self) -> None:
        """Inicializa o analisador com uma lista vazia de cursos."""
        self.cursos: List[Curso] = []
    
    async def extrair_cursos(self, url: str):
        """
        Extrai informações de cursos de uma URL específica.
        
        Args:
            url (str): URL da página a ser analisada
            
        Returns:
            Dict[str, Any]: Dicionário contendo:
                - conteudo: texto extraído da página
                - links: lista de links encontrados
                - imagens: lista de URLs de imagens
        """
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    excluded_tags=['nav', 'footer']
                )
            )
            
            texto = result.markdown_v2.raw_markdown
            
            # Aqui você poderia usar regex ou outras técnicas
            # para extrair informações específicas dos cursos
            
            return {
                "conteudo": texto,
                "links": result.links,
                "imagens": result.media['images']
            }


    async def extrair_imagens(self, url: str):
        """
        Extrai apenas as URLs das imagens de uma página web.
        
        Args:
            url (str): URL da página a ser analisada
                
        Returns:
            List[str]: Lista com URLs das imagens encontradas
        """
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    excluded_tags=['nav', 'footer']
                )
            )
            return result.media['images']


    # Função main atualizada
async def main():
    """
    Função principal que executa o scraping de imagens e salva os resultados.
    """
    analisador = AnalisadorCursos()
    imagens = await analisador.extrair_imagens(
        "https://www.kidocode.com/degrees/technology"
    )
    
    # Salvar apenas as imagens em JSON
    import json
    with open('imagens_cursos.json', 'w') as f:
        json.dump({"imagens": imagens}, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())