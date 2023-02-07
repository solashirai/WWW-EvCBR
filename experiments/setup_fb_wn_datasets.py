import rdflib
from rdflib import URIRef
from utils import *
import os
import math
import time
import random

# set random seed for more consistent testing
random.seed(100)

relations = dict()

fb_in_dir = (DATA_DIR / "fb15k-237/Release/").resolve()
fb_out_dir = (DATA_DIR / "pp_fb15k/").resolve()
wn_in_dir = (DATA_DIR / "wn18rr/text/").resolve()
wn_out_dir = (DATA_DIR / "pp_wn18rr/").resolve()

def preprocess_data(in_dir, out_dir, tv_n, real_tv_n:int=0):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    g = rdflib.Graph()

    # original train/valid/test split is intended for regular training and testing
    # so we want to load it all into a graph and re-select the entities to make it suitable for
    # testing inductive link prediction
    for fname in ["train.txt", "valid.txt", "test.txt"]:
        with open((in_dir / fname).resolve(), "r") as f:
            for line in f:
                trip = line.strip().split("\t")
                if trip:
                    g.add((URIRef(trip[0]), URIRef(trip[1]), URIRef(trip[2])))
                    if URIRef(trip[1]) not in relations:
                        relations[URIRef(trip[1])] = len(relations)

    # split into train/valid/test by selecting entities rather than selecting triples.
    # test and valid entities must have at least one outgoing edge and one incoming edge.
    # the incoming edges will be stored separately from the test triples, since they provide us with information about
    # what incoming relation we will perform the test on. this will have to be done in a custom testing script for each
    # model being evaluated.
    # for the moment, we'll aim to get 2000 entities each in the validation/test entity sets (roughly 5% of all entities)

    all_ents = set(g.all_nodes())
    testvalid_options = all_ents
    train_ents = set() # explicitly add entities to the train set if they can't be selected for valid/test
    print(f"{len(all_ents)} unique entities.")

    test_ents = set()
    valid_ents = set()

    def check_ent_is_isolated(src_ent, target_ent, test_ents, valid_ents):
        connection_count = 0
        connected_ents = set()
        for tt in g.triples((None, None, target_ent)):
            if tt[0] != src_ent and tt[0] not in test_ents and tt[0] not in valid_ents and tt[0] not in connected_ents:
                connection_count += 1
                connected_ents.add(tt[0])
        for tt in g.triples((target_ent, None, None)):
            if tt[2] != src_ent and tt[2] not in test_ents and tt[2] not in valid_ents and tt[2] not in connected_ents:
                connection_count += 1
                connected_ents.add(tt[2])
        if connection_count < 1:
            return True
        return False

    start_time = time.time()
    for target_set in [test_ents, valid_ents]:
        while len(target_set) < tv_n:
            ent_choice = random.sample(testvalid_options, k=1)[0]

            # prevent self-loops for anything in the valid/test set since methods wont be able to handle them
            if len(list(g.triples((ent_choice, None, ent_choice)))) > 0:
                train_ents.add(ent_choice)
                testvalid_options.remove(ent_choice)
                continue
            # the test/valid entity must have at least one incoming and outgoing edge
            if not len(list(g.triples((None, None, ent_choice)))) or not len(list(g.triples((ent_choice, None, None)))):
                train_ents.add(ent_choice)
                testvalid_options.remove(ent_choice)
                continue

            good_choice = False
            # ensure there's a non-test/valid entity that is incoming to the test ent
            for t in g.triples((None, None, ent_choice)):
                if ent_choice in train_ents:
                    continue
                # if an incoming node is only reachable from the chosen ent, then we need to ensure this ent
                # isnt in the valid/test set
                if check_ent_is_isolated(ent_choice, t[0], test_ents, valid_ents):
                    train_ents.add(ent_choice)
                    testvalid_options.remove(ent_choice)
                    continue

                if t[0] in train_ents or (t[0] not in test_ents and t[0] not in valid_ents):
                    good_choice = True
            if ent_choice in train_ents:
                continue
            if not good_choice:
                train_ents.add(ent_choice)
                testvalid_options.remove(ent_choice)
                continue

            # for outgoing edges, ensure that there's at least one other path to reach it besides the entity currently being
            # chosen. if there isn't, the models will have no way of being able to make any correct prediction about it.
            for t in g.triples((ent_choice, None, None)):
                if ent_choice in train_ents:
                    continue
                if check_ent_is_isolated(ent_choice, t[2], test_ents, valid_ents):
                    train_ents.add(ent_choice)
                    testvalid_options.remove(ent_choice)
                    continue
            if ent_choice in train_ents:
                continue

            # if ent_choice hasn't been added to train_ents by this point, add it to the valid or test set then add
            # all incoming/outgoing entities to the train set
            target_set.add(ent_choice)
            testvalid_options.remove(ent_choice)
            for t in g.triples((ent_choice, None, None)):
                train_ents.add(t[2])
                testvalid_options -= {t[2]}
            for t in g.triples((None, None, ent_choice)):
                train_ents.add(t[0])
                testvalid_options -= {t[0]}

    train_ents = train_ents.union(testvalid_options)
    if real_tv_n != 0:
        test_ents = set(random.sample(test_ents, real_tv_n))
        valid_ents = set(random.sample(valid_ents, real_tv_n))

    end_time = time.time()
    print(f"time taken to select {tv_n} test/valid entities: {end_time-start_time}")
    print("sanity checking train/test/valid lengths and number of entities overlapping (there should be 0 overlaps)")
    print(len(train_ents), len(test_ents), len(valid_ents), len(train_ents.intersection(test_ents)),
          len(train_ents.intersection(valid_ents)), len(valid_ents.intersection(test_ents)))
    print(f"{len(g.all_nodes())} total entities")

    test_triples = []
    test_connection_triples = []
    valid_triples = []
    valid_connection_triples = []
    train_triples = []

    connection_tail_not_in_train = 0

    for t in g:
        if t[0] in train_ents:
            if t[2] in test_ents:
                test_connection_triples.append(t)
            elif t[2] in valid_ents:
                valid_connection_triples.append(t)
            elif t[2] in train_ents:
                train_triples.append(t)
        elif t[0] in valid_ents:
            valid_triples.append(t)
            if t[2] not in train_ents:
                connection_tail_not_in_train += 1
        elif t[0] in test_ents:
            test_triples.append(t)
            if t[2] not in train_ents:
                connection_tail_not_in_train += 1
    print(f"sanity checking: {connection_tail_not_in_train} test tail entities missing from training set")

    print(f"number of triples in train/valid/test sets: {len(train_triples)}, {len(valid_triples)}, {len(test_triples)}")
    print(f"{len(valid_connection_triples)} triples for incoming connections to validation set ents, "
          f"{len(test_connection_triples)} for test.")

    with open((out_dir / "train.txt").resolve(), "w") as f:
        for t in train_triples:
            f.write(f"{t[0]}\t{t[1]}\t{t[2]}\n")
    with open((out_dir / "valid.txt").resolve(), "w") as f:
        for t in valid_triples:
            f.write(f"{t[0]}\t{t[1]}\t{t[2]}\n")
    with open((out_dir / "valid_connections.txt").resolve(), "w") as f:
        for t in valid_connection_triples:
            f.write(f"{t[0]}\t{t[1]}\t{t[2]}\n")
    with open((out_dir / "test.txt").resolve(), "w") as f:
        for t in test_triples:
            f.write(f"{t[0]}\t{t[1]}\t{t[2]}\n")
    with open((out_dir / "test_connections.txt").resolve(), "w") as f:
        for t in test_connection_triples:
            f.write(f"{t[0]}\t{t[1]}\t{t[2]}\n")

    # also set up a entities/relations dict file, since some models need it to run.
    # note that entities in the valid/test set aren't included in this dict, since we should be doing inductive LP

    entities = dict()
    for e in train_ents:
        entities[e] = len(entities)

    rev_entities = {v:k for k,v in entities.items()}
    rev_relations = {v:k for k,v in relations.items()}

    with open((out_dir / "entities.dict").resolve(), "w") as f:
        for i in range(len(rev_entities)):
            f.write(f"{i}\t{rev_entities[i]}\n")
    with open((out_dir / "relations.dict").resolve(), "w") as f:
        for i in range(len(rev_relations)):
            f.write(f"{i}\t{rev_relations[i]}\n")

print("starting preprocessing WN18RR dataset")
preprocess_data(in_dir=wn_in_dir, out_dir=wn_out_dir, tv_n=1000)
print("finished preprocessing WN18RR dataset")
print("starting preprocessing FB15k-237 dataset")
preprocess_data(in_dir=fb_in_dir, out_dir=fb_out_dir, tv_n=500)
print("finished preprocessing FB15k-237 dataset")