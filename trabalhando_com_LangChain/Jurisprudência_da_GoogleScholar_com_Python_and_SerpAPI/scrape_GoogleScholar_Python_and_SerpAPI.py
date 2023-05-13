"""
Data Scientist.: PhD.Eddy Giusepe Chirinos Isidro

Este estudo está baseado na publicação de Dmitriy Zub 🥸. A publicação foi no Medium.


Scrape Google Scholar Case Law Results to CSV in Python and SerpApi
===================================================================
Um guia abrangente sobre como copiar os resultados da Jurisprudência do Google Scholar para CSV em Python e SerpApi.
Link de estudo --> https://python.plainenglish.io/scrape-google-scholar-case-law-results-to-csv-in-python-and-serpapi-af4777fcfc87


Instalação das seguintes bibliotecas
------------------------------------

$ pip install pandas google-search-results
"""
import os # é usado para retornar o valor da variável de ambiente de chave da API SerpApi.
from serpapi import GoogleSearch
from urllib.parse import urlsplit, parse_qsl # urlib --> será usado no processo de paginação.
import pandas as pd


# Isto é quando usas o arquivo .env: 
from dotenv import load_dotenv
#print('Carregando a minha chave Key: ', load_dotenv())
load_dotenv()
Eddy_API_KEY_SerpApi = os.environ["SERPAPI_API_KEY"]

def case_law_results():
    print("Extracting case law results..")
    params = {
        "api_key": Eddy_API_KEY_SerpApi, #os.getenv("API_KEY"),  # SerpApi API key
        "engine": "google_scholar",       # Google Scholar search results
        "q": "minecraft education",      # search query
        "hl": "en",                       # language
        "start": "0",                     # first page
        "as_sdt": "6"                     # case law results. Wierd, huh? Try without it.
    }
    """
    'as_sdt' é usado para determinar e filtrar quais tribunais são alvo de uma chamada de API. 
    Consulte "support SerpApi Google Scholar Courts" (https://serpapi.com/google-scholar-courts) de SerpApi ou 
    selecione "select courts on Google Scholar" (https://scholar.google.com/scholar_courts?q=blizzard&hl=en&as_sdt=2006) 
    e passe para o parâmetro 'as_sdt'.
    """
    search = GoogleSearch(params)

    case_law_results_data = []

    while True:
        results = search.get_dict()
        if "error" in results:
            break

    print(f"Currently extracting page #{results.get('serpapi_pagination', {}).get('current')}..")

    """
    Extraia os resultados em um loop for e trate as exceções:
    """
    for result in results["organic_results"]:
        title = result.get("title")
        publication_info_summary = result["publication_info"]["summary"]
        result_id = result.get("result_id")
        link = result.get("link")
        result_type = result.get("type")
        snippet = result.get("snippet")

        try:
          file_title = result["resources"][0]["title"]
        except: file_title = None

        try:
          file_link = result["resources"][0]["link"]
        except: file_link = None

        try:
          file_format = result["resources"][0]["file_format"]
        except: file_format = None

        cited_by_count = result.get("inline_links", {}).get("cited_by", {}).get("total", {})
        cited_by_id = result.get("inline_links", {}).get("cited_by", {}).get("cites_id", {})
        cited_by_link = result.get("inline_links", {}).get("cited_by", {}).get("link", {})
        total_versions = result.get("inline_links", {}).get("versions", {}).get("total", {})
        all_versions_link = result.get("inline_links", {}).get("versions", {}).get("link", {})
        all_versions_id = result.get("inline_links", {}).get("versions", {}).get("cluster_id", {})
        
        case_law_results_data.append({
          "page_number": results['serpapi_pagination']['current'],
          "position": result["position"] + 1,
          "result_type": result_type,
          "title": title,
          "link": link,
          "result_id": result_id,
          "publication_info_summary": publication_info_summary,
          "snippet": snippet,
          "cited_by_count": cited_by_count,
          "cited_by_link": cited_by_link,
          "cited_by_id": cited_by_id,
          "total_versions": total_versions,
          "all_versions_link": all_versions_link,
          "all_versions_id": all_versions_id,
          "file_format": file_format,
          "file_title": file_title,
          "file_link": file_link
        })

        if "next" in results.get("serpapi_pagination", {}):
            search.params_dict.update(dict(parse_qsl(urlsplit(results["serpapi_pagination"]["next"]).query)))
        else:
           break

    return case_law_results_data

print("Agora salvamos os Dados: ")
def save_case_law_results_to_csv():
    print("Waiting for case law results to save..")
    pd.DataFrame(data=case_law_results()).to_csv("google_scholar_case_law_results.csv", encoding="utf-8", index=False)
    print("Case Law Results Saved.")
    

print("Vamos ver como é salvo: ")
save_case_law_results_to_csv()






