from rdflib import Namespace
from pathlib import Path
import os
import pickle


WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

WDT_SUBCLASSOF = WDT["P279"]
WDT_INSTANCEOF = WDT["P31"]
WDT_HASEFFECT = WDT["P1542"]
WDT_DIFFERENTFROM = WDT["P1889"]

ROOT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = Path.resolve(ROOT_DIR / "data")
DEMO_DATA_DIR = Path.resolve(ROOT_DIR / "demo_data")

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
