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
from typing import Dict, List
import time
from utils import *


def load_indexes(data_dir):
    entity_to_index = dict()
    relation_to_index = dict()
    with open((data_dir / "entities.dict").resolve(), "r") as f:
        for line in f:
            content = line.strip().split("\t")
            entity_to_index[URIRef(content[1])] = int(content[0])
    with open((data_dir / "relations.dict").resolve(), "r") as f:
        for line in f:
            content = line.strip().split("\t")
            relation_to_index[URIRef(content[1])] = int(content[0])
    return entity_to_index, relation_to_index

def vectorize_graph(in_dir: Path, out_dir: Path, do_outgoing_edges: bool = False,  do_two_hop: bool = False, do_superclasses: List[URIRef] = None):
    # construct a 3d matrix of the form (s, p, o) to support faster calculation of adjacencies/paths
    print("initializing mapping of URIs to indexes")
    e2i, r2i = load_indexes(in_dir)
    print(f"{len(e2i)} entities, {len(r2i)} relations loaded.")
    formatted_graph = defaultdict(lambda: defaultdict(lambda: set()))
    formatted_graph_inverse = defaultdict(lambda: defaultdict(lambda: set()))

    print("starting conversion of triples into a dictionary of dictionaries.")
    ent_rel_matrix = dok_matrix((len(e2i.keys()), len(r2i.keys())))
    with open((in_dir / "train.txt").resolve(), "r") as f:
        for line in f:
            triple = line.strip().split("\t")
            s,p,o = URIRef(triple[0]), URIRef(triple[1]), URIRef(triple[2])
            formatted_graph[e2i[s]][r2i[p]].add(e2i[o])
            formatted_graph_inverse[e2i[o]][r2i[p]].add(e2i[s])
            ent_rel_matrix[e2i[s],r2i[p]] = 1

    ent_rel_matrix = ent_rel_matrix.tocsr()
    # convert back to normal dictionaries to enable pickling
    for k,v in formatted_graph.items():
        formatted_graph[k] = dict(v)
    formatted_graph = dict(formatted_graph)
    for k,v in formatted_graph_inverse.items():
        formatted_graph_inverse[k] = dict(v)
    formatted_graph_inverse = dict(formatted_graph_inverse)

    print("starting subclass matrix creation...")
    print("this code is not very well optimized.")
    print("for experiments on wn18rr/fb15k datasets, this isn't actually a matrix of subclasses, but rather of "
          "other entities that are connected by outgoing edges.")
    subclass_matrix = dok_matrix((len(e2i.keys()), len(e2i.keys())))
    subclass_counts = np.zeros((len(e2i.keys()),))
    subclass_totals = 0

    if do_superclasses:
        print("doing superclass based similarity")
        def recursive_superclasses(formatted_graph, r2i, step, seen):
            outputs = set()
            for s in step:
                outputs.add(s)
                if s in seen:
                    continue
                else:
                    seen.add(s)

                    sp = set()
                    for r in do_superclasses:
                        sp = sp.union(formatted_graph.get(s, {}).get(r2i[r], set()))
                    outputs = outputs.union(recursive_superclasses(formatted_graph, r2i, sp, seen))
            return outputs

        for ent in e2i.keys():
            if ent not in e2i.keys():
                continue
            subclass_totals += 1
            subclass_matrix[e2i[ent], e2i[ent]] = 1
            subclass_counts[e2i[ent]] += 1
            sp = set()
            for r in do_superclasses:
                sp = sp.union(formatted_graph.get(e2i[ent], {}).get(r2i[r], set()))
            all_super = recursive_superclasses(formatted_graph, r2i, sp, set())

            for s in all_super:
                subclass_matrix[e2i[ent], s] = 1
                subclass_counts[s] += 1
    if do_outgoing_edges:
        print("doing outgoing edge-based similarity")
        if do_two_hop:
            print("computing 2-hop outgoing edges")
        for ent in e2i.keys():
            subclass_totals += 1
            subclass_matrix[e2i[ent], e2i[ent]] = 1
            subclass_counts[e2i[ent]] += 1

            all_super = set()
            # for datasets that don't contain subclass relations / sublclass relations aren't common,
            # instead just collect all incoming/outgoing connected entities in a 2hop neighborhood
            for r in formatted_graph.get(e2i[ent], {}).keys():
                for o in formatted_graph[e2i[ent]][r]:
                    all_super.add(o)

                    if do_two_hop:
                        for r2 in formatted_graph.get(o, {}).keys():
                            for o2 in formatted_graph[o][r2]:
                                all_super.add(o2)
                        # for r2 in formatted_graph_inverse.get(o, {}).keys():
                        #     for o2 in formatted_graph_inverse[o][r2]:
                        #         all_super.add(o2)

            for r in formatted_graph_inverse.get(e2i[ent], {}).keys():
                for o in formatted_graph_inverse[e2i[ent]][r]:
                    all_super.add(o)

                    if do_two_hop:
                        for r2 in formatted_graph.get(o, {}).keys():
                            for o2 in formatted_graph[o][r2]:
                                all_super.add(o2)
                        # for r2 in formatted_graph_inverse.get(o, {}).keys():
                        #     for o2 in formatted_graph_inverse[o][r2]:
                        #         all_super.add(o2)
            if len(all_super) == 0:
                print("!?!?!?!!?", ent, e2i[ent])

            for s in all_super:
                subclass_matrix[e2i[ent], s] = 1
                subclass_counts[s] += 1
    subclass_matrix = subclass_matrix.tocsr()

    for i in range(subclass_counts.shape[0]):
        if subclass_counts[i] != 0:
            subclass_counts[i] = np.log(subclass_totals/subclass_counts[i])

    print("saving preprocessing results")

    with open(Path.resolve(out_dir / "formatted_graph.pkl"), "wb") as f:
        pickle.dump(formatted_graph, f)
    with open(Path.resolve(out_dir / "formatted_graph_inv.pkl"), "wb") as f:
        pickle.dump(formatted_graph_inverse, f)
    with open(Path.resolve(out_dir / "subclass_matrix.pkl"), "wb") as f:
        pickle.dump(subclass_matrix, f)
    with open(Path.resolve(out_dir / "subclass_idf.pkl"), "wb") as f:
        pickle.dump(subclass_counts, f)
    with open(Path.resolve(out_dir / "ent_rel_matrix.pkl"), "wb") as f:
        pickle.dump(ent_rel_matrix, f)
    with open(Path.resolve(out_dir / "e2i.pkl"), "wb") as f:
        pickle.dump(e2i, f)
    with open(Path.resolve(out_dir / "r2i.pkl"), "wb") as f:
        pickle.dump(r2i, f)
    return formatted_graph, formatted_graph_inverse

def main(args):
    if args.process_wn:
        wn_input_dir = (DATA_DIR / args.wn_input).resolve()
        wn_output_dir = (DATA_DIR / args.wn_output).resolve()
        if not os.path.isdir(wn_output_dir):
            os.mkdir(wn_output_dir)

    if args.process_fb:
        fb_input_dir = (DATA_DIR / args.fb_input).resolve()
        fb_output_dir = (DATA_DIR / args.fb_output).resolve()
        if not os.path.isdir(fb_output_dir):
            os.mkdir(fb_output_dir)

    if args.process_wiki:
        wiki_input_dir = (DATA_DIR / args.wiki_input).resolve()
        wiki_output_dir = (DATA_DIR / args.wiki_output).resolve()
        if not os.path.isdir(wiki_output_dir):
            os.mkdir(wiki_output_dir)

    print(f"starting vectorization and preprocessing")
    if args.process_wn:
        fg, fgi = vectorize_graph(in_dir=wn_input_dir, out_dir=wn_output_dir, do_outgoing_edges=True, do_two_hop=True,
                               # do_superclasses=[URIRef('_hypernym')])
                               do_superclasses=[])
    else:
        print(f"skipping WN18RR")
    if args.process_fb:
        print("starting FB15k preprocessing")
        fg, fgi = vectorize_graph(in_dir=fb_input_dir, out_dir=fb_output_dir, do_outgoing_edges=True, do_two_hop=True)
    else:
        print("skipping FB")

    if args.process_wiki:
        print("starting wikidata preprocessing")
        fg, fgi = vectorize_graph(in_dir=wiki_input_dir, out_dir=wiki_output_dir, do_outgoing_edges=False, do_two_hop=False,
                              do_superclasses=[WDT_SUBCLASSOF, WDT_INSTANCEOF])
    else:
        print("skipping wikidata")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess an rdflib graph into a matrix format")
    parser.add_argument("--process_wn",action='store_true', default=False)
    parser.add_argument("--process_fb",action='store_true', default=False)
    parser.add_argument("--process_wiki",action='store_true', default=False)
    parser.add_argument("--wn_input", type=str, default="pp_wn18rr")
    parser.add_argument("--wn_output", type=str, default="evcbr_pp_wn18rr")
    parser.add_argument("--fb_input", type=str, default="pp_fb15k")
    parser.add_argument("--fb_output", type=str, default="evcbr_pp_fb15k")
    parser.add_argument("--wiki_input", type=str, default="pp_wiki")
    parser.add_argument("--wiki_output", type=str, default="evcbr_pp_wiki")
    args = parser.parse_args()

    main(args)