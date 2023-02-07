import rdflib
from rdflib import Graph
import pickle
import numpy as np
import json
import argparse
from pathlib import Path
from scipy.sparse import dok_matrix
from collections import defaultdict
from rdflib import URIRef
import os
import random
from typing import Dict
import time
from utils import *

random.seed(100)

def create_uri_to_index(kg: Graph):
    entity_to_index = dict()
    relation_to_index = dict()
    for (s,p,o) in kg:
        if p not in relation_to_index:
            relation_to_index[p] = len(relation_to_index)
        if isinstance(s, URIRef) and s not in entity_to_index:
            entity_to_index[s] = len(entity_to_index)
        if isinstance(o, URIRef) and o not in entity_to_index:
            entity_to_index[o] = len(entity_to_index)
    return entity_to_index, relation_to_index

def main(args):
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    input_kg = (DATA_DIR/args.input_kg).resolve()
    subclass_kg = (DATA_DIR/args.subclass_kg).resolve()

    out_dir = (DATA_DIR/args.out_dir).resolve()
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    print(f"Loading graphs")
    kg = rdflib.Graph()
    kg.parse(input_kg)

    with open(RELEVANT_EVENTS_DATA, 'rb') as f:
        relevant_events = pickle.load(f)
    relevant_events = set(relevant_events)

    eval_options = set()
    # to expand the amount of available data, convert all cause/effect types to just be "has effect" relations
    for effect_prop in EFFECT_PROPERTIES:
        remove_trips = []
        add_trips = []
        for s,o in kg.subject_objects(predicate=effect_prop):
            remove_trips.append((s, effect_prop, o))
            add_trips.append((s, WDT_HASEFFECT, o))
        for rt in remove_trips:
            kg.remove(rt)
        for at in add_trips:
            kg.add(at)
    for cause_prop in CAUSE_PROPERTIES:
        remove_trips = []
        add_trips = []
        for o, s in kg.subject_objects(predicate=cause_prop):
            remove_trips.append((o, cause_prop, s))
            add_trips.append((s, WDT_HASEFFECT, o))
        for rt in remove_trips:
            kg.remove(rt)
        for at in add_trips:
            kg.add(at)

    for s,o in kg.subject_objects(predicate=WDT_HASEFFECT):
        for cause_type in kg.objects(subject=s, predicate=WDT_INSTANCEOF):
            for o_ in kg.objects(subject=o, predicate=WDT_INSTANCEOF):
                if cause_type in relevant_events and o_ in relevant_events and \
                        s not in relevant_events and o not in relevant_events:
                    eval_options.add(cause_type)
                    break

    cause_effect_truth = set()
    for target_event_type in eval_options:
        eval_cases = list(kg.subjects(predicate=WDT_INSTANCEOF, object=target_event_type))
        for sc in eval_cases:
            for effect in kg.objects(subject=sc, predicate=WDT_HASEFFECT):
                effect_types = set(kg.objects(subject=effect, predicate=WDT_INSTANCEOF))
                if len(effect_types.intersection(relevant_events)) > 0 and effect not in relevant_events:
                    cause_effect_truth.add((sc, effect))
    print(f"{len(eval_options)} different event classes to perform evaluation.")
    print(f"{len(cause_effect_truth)} cause-effect pairs")

    # cleaning the kg
    print(f"Initial triples: {len(kg)}")
    rem_lit = set()
    for (s, p, o) in kg:
        if isinstance(o, rdflib.Literal):
            rem_lit.add(o)
    for lit in rem_lit:
        kg.remove((None, None, lit))
    print(f"literals removed, {len(kg)} triples present.")
    rem_trip = set()
    for (s, p, o) in kg.triples((None, WDT_DIFFERENTFROM, None)):
        rem_trip.add((s, p, o))
    for rt in rem_trip:
        kg.remove(rt)
    print(f"Removed 'differentFrom' relations, {len(kg)} triples present")
    rem_wm = set()
    print(f"removing URIs that look like media/images")
    for (s, p, o) in kg:
        if "wikimedia.org" in o:
            rem_wm.add(o)
        elif "wikimedia" in o or ".png" in o or ".jpg" in o:
            rem_wm.add(o)
    for ent in rem_wm:
        kg.remove((None, None, ent))
    print(f"wikimedia URI removed, {len(kg)} triples present.")

    test_causeeffect_choices = random.sample(cause_effect_truth, k=100)
    test_causes = set([tup[0] for tup in test_causeeffect_choices])
    test_effects = set([tup[1] for tup in test_causeeffect_choices])
    for (cause, effect) in test_causeeffect_choices:
        if effect in test_causes:
            if effect in test_effects:
                test_effects.remove(effect)
        elif cause in test_effects:
            if cause in test_effects:
                test_effects.remove(cause)
    print(f'{len(test_effects)} effects selected for testing')


    print(f"removing nodes with only 1 connection")
    remove_nodes = []
    for n in kg.all_nodes():
        conn_count = 0
        for s,p, in kg.subject_predicates(object=n):
            conn_count += 1
            if conn_count > 1:
                break
        for p,o in kg.predicate_objects(subject=n):
            conn_count += 1
            if conn_count > 1:
                break
        if conn_count <= 1:
            remove_nodes.append(n)
    while remove_nodes:
        for rn in remove_nodes:
            kg.remove((None, None, rn))
            kg.remove((rn, None, None))
        remove_nodes = []
        for n in kg.all_nodes():
            conn_count = 0
            for s, p, in kg.subject_predicates(object=n):
                conn_count += 1
                if conn_count > 1:
                    break
            for p, o in kg.predicate_objects(subject=n):
                conn_count += 1
                if conn_count > 1:
                    break
            if conn_count <= 1:
                remove_nodes.append(n)
    print(f"finished cleaning isolated nodes, {len(kg)} triples present.")

    print("setting up triples")
    test_triples = []
    test_connection_triples = []
    valid_triples = []
    valid_connection_triples = []
    train_ents = set()
    train_relations = set()
    train_triples = []

    training_kg = rdflib.Graph()
    for t in kg:
        if t[0] in test_effects:
            test_triples.append(t)
        elif t[2] in test_effects:
            # only add triples incoming to the effect if its from a cause
            if t[0] in test_causes:
                test_connection_triples.append(t)
        else:
            train_triples.append(t)
            train_ents.add(t[0])
            train_relations.add(t[1])
            train_ents.add(t[2])
            training_kg.add(t)

    # moving on to saving and preprocessing training data
    kg = training_kg

    with open((out_dir / "train.txt").resolve(), "w", encoding='utf-8') as f:
        for t in train_triples:
            f.write(f"{t[0]}\t{t[1]}\t{t[2]}\n")
    with open((out_dir / "test.txt").resolve(), "w", encoding='utf-8') as f:
        for t in test_triples:
            f.write(f"{t[0]}\t{t[1]}\t{t[2]}\n")
    with open((out_dir / "test_connections.txt").resolve(), "w", encoding='utf-8') as f:
        for t in test_connection_triples:
            f.write(f"{t[0]}\t{t[1]}\t{t[2]}\n")

    entities = dict()
    for e in train_ents:
        entities[e] = len(entities)
    relations = dict()
    for r in train_relations:
        relations[r] = len(relations)
    rev_entities = {v: k for k, v in entities.items()}
    rev_relations = {v: k for k, v in relations.items()}
    with open((out_dir / "entities.dict").resolve(), "w", encoding='utf-8') as f:
        for i in range(len(rev_entities)):
            f.write(f"{i}\t{rev_entities[i]}\n")
    with open((out_dir / "relations.dict").resolve(), "w", encoding='utf-8') as f:
        for i in range(len(rev_relations)):
            f.write(f"{i}\t{rev_relations[i]}\n")

    subkg = rdflib.Graph()
    subkg.parse(subclass_kg)

    kg += subkg
    print(f"Finished loading.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess an rdflib graph into a matrix format")
    parser.add_argument("--input_kg", type=str, default="wikidata_cc_full_3_hop.ttl")
    parser.add_argument("--subclass_kg", type=str, default="wikidata_subclasses.ttl")
    parser.add_argument("--out_dir", type=str, default="pp_wiki/")
    args = parser.parse_args()

    main(args)