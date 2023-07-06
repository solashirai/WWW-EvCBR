from rdflib import Namespace, URIRef
import rdflib
from pathlib import Path
import os
import pickle
from typing import List, Dict, Set
import requests
from collections import defaultdict


WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

WDT_SUBCLASSOF = WDT["P279"]
WDT_INSTANCEOF = WDT["P31"]
WDT_HASEFFECT = WDT["P1542"]
WDT_DIFFERENTFROM = WDT["P1889"]

ROOT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = Path.resolve(ROOT_DIR / "data")
DEMO_DATA_DIR = Path.resolve(ROOT_DIR / "demo_data")

RDFS_LABEL = rdflib.namespace.RDFS.label

EFFECT_PROPERTIES = [WDT["P1542"], WDT["P1536"], WDT["P1537"]]
CAUSE_PROPERTIES = [WDT["P828"], WDT["P1478"], WDT["P1479"]]

RELEVANT_EVENTS_DATA = Path.resolve(DATA_DIR / "relevant_event_types.pkl")


def add_haseffect_relations(kg):
    with open(RELEVANT_EVENTS_DATA, 'rb') as f:
        rel_events = set(pickle.load(f))

    for p in EFFECT_PROPERTIES:
        if p == WDT_HASEFFECT:
            continue
        for s,o in kg.subject_objects(predicate=p):
            s_types = set(kg.objects(subject=s, predicate=WDT_INSTANCEOF))
            o_types = set(kg.objects(subject=o, predicate=WDT_INSTANCEOF))
            if len(s_types.intersection(rel_events)) >= 1 and len(o_types.intersection(rel_events)) >= 1:
                kg.add((s, WDT_HASEFFECT, o))

    for p in CAUSE_PROPERTIES:
        if p == WDT_HASEFFECT:
            continue
        for s,o in kg.subject_objects(predicate=p):
            s_types = set(kg.objects(subject=s, predicate=WDT_INSTANCEOF))
            o_types = set(kg.objects(subject=o, predicate=WDT_INSTANCEOF))
            if len(s_types.intersection(rel_events)) >= 1 and len(o_types.intersection(rel_events)) >= 1:
                kg.add((o, WDT_HASEFFECT, s))


def collect_wikidata_labels(uri_list: List[URIRef]) -> Dict[URIRef, str]:
    valid_uris = ["wd:"+qid for u in uri_list if (qid:=u.split("/")[-1])[0] == "Q" and qid[1:].isnumeric()]
    sparql_str = f"""
    SELECT ?u ?uLabel WHERE {{
        VALUES ?u {{ {' '.join(valid_uris)} }}
       SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en" .
       }}
    }}
    """
    print("Collecting entity labels from Wikidata")
    url = "https://query.wikidata.org/sparql"
    res_get = requests.get(url, params={'format': 'json',
                                        'query': sparql_str})
    data = res_get.json()
    out_dict = defaultdict(lambda: "")
    for row in data['results']['bindings']:
        out_dict[URIRef(row['u']['value'])] = row['uLabel']['value']
    return out_dict


def collect_wikidata_property_labels(uri_list: List[URIRef]) -> Dict[URIRef, str]:
    valid_uris = ["wd:"+qid for u in uri_list if (qid:=u.split("/")[-1])[0] == "P" and qid[1:].isnumeric()]
    sparql_str = f"""
    SELECT ?u ?uLabel WHERE {{
        VALUES ?u {{ {' '.join(valid_uris)} }}
       SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en" .
       }}
    }}
    """
    print("Collecting property labels from Wikidata")
    url = "https://query.wikidata.org/sparql"
    res_get = requests.get(url, params={'format': 'json',
                                        'query': sparql_str})
    data = res_get.json()
    out_dict = defaultdict(lambda: "")
    for row in data['results']['bindings']:
        out_dict[row['u']['value'].split("/")[-1]] = row['uLabel']['value']
    return out_dict