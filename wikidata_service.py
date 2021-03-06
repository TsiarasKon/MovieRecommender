import time

import pandas as pd
import requests
from http import HTTPStatus

from tqdm import tqdm

wikidata_endpoint = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"
default_retry_after = 0.5


def response_to_df(response_list):
    tconst_list = []
    series_list = []
    distributors_list = []
    subject_list = []
    for entity in response_list:
        if not entity['series']['value'] and not entity['distributors']['value'] and not entity['subject']['value']:
            continue
        tconst_list.append(entity['tconst']['value'])
        series_list.append(entity['series']['value'])
        distributors_list.append(entity['distributors']['value'])
        subject_list.append(entity['subject']['value'])
    return pd.DataFrame({'tconst': tconst_list, 'series': series_list, 'distributors': distributors_list, 'subject': subject_list})


def query_wikidata(imdb_ids_list):
    # query = """SELECT ?movieid ?movieLabel
    #   (group_concat(distinct ?seriesLabel; separator="; ") as ?series)
    #   (group_concat(distinct ?distributorsLabel; separator="; ") as ?distributors)
    #   (group_concat(distinct ?subjectLabel; separator="; ") as ?subject)
    #   (group_concat(distinct ?boxofficeLabel; separator="; ") as ?boxoffice)
    # WHERE {
    #   VALUES ?movieid { "%s" }
    #   ?movie wdt:P345 ?movieid .
    #   OPTIONAL { ?movie wdt:P179 ?series . }
    #   OPTIONAL { ?movie wdt:P750 ?distributors . }
    #   OPTIONAL { ?movie wdt:P921 ?subject . }
    #   OPTIONAL { ?movie wdt:P2142 ?boxoffice . }
    #   SERVICE wikibase:label {
    #     bd:serviceParam wikibase:language "en".
    #     ?movie rdfs:label ?movieLabel .
    #     ?series rdfs:label ?seriesLabel .
    #     ?distributors rdfs:label ?distributorsLabel .
    #     ?subject rdfs:label ?subjectLabel .
    #     ?boxoffice rdfs:label ?boxofficeLabel .
    #   }
    # }
    # GROUP BY ?movieid ?movieLabel""" % '" "'.join(imdb_ids_list)

    query = """SELECT ?tconst
      (group_concat(distinct ?seriesID; separator="; ") as ?series)
      (group_concat(distinct ?distributorsID; separator="; ") as ?distributors)
      (group_concat(distinct ?subjectID; separator="; ") as ?subject)
    WHERE {
      VALUES ?tconst { "%s" }
      ?movieID wdt:P345 ?tconst .
      OPTIONAL { ?movieID wdt:P179 ?seriesID . }
      OPTIONAL { ?movieID wdt:P750 ?distributorsID . }
      OPTIONAL { ?movieID wdt:P921 ?subjectID . }
    }
    GROUP BY ?tconst""" % '" "'.join(imdb_ids_list)

    response = requests.get(wikidata_endpoint, params={'format': 'json', 'query': query})
    if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        time.sleep(int(response.headers['Retry-After']) if 'Retry-After' in response.headers else default_retry_after)
        response = requests.get(wikidata_endpoint, params={'format': 'json', 'query': query})
    if response.status_code != HTTPStatus.OK:
        print(f"Encountered error querying wikidata: {response.status_code}")
        return response_to_df([])
    return response_to_df(response.json()['results']['bindings'])


def get_all_from_wikidata(imdb_id_list):
    print(len(imdb_id_list))
    wikidata_df = response_to_df([])    # init empty df with desired columns
    for i in tqdm(range(0, len(imdb_id_list), 200)):
        wikidata_df = pd.concat([wikidata_df, query_wikidata(imdb_id_list[i:min(i+200, len(imdb_id_list))])])
    wikidata_df.set_index('tconst', inplace=True)
    return wikidata_df
