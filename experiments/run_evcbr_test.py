import os
from frozendict import frozendict
import rdflib
from utils import *
from evcbr import EvCBR
from rdflib import Graph, Literal, URIRef
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
import numpy as np
import argparse
from collections import defaultdict
import time
import multiprocessing as mp
import pickle
from typing import Tuple, List
import networkx as nx
import random


MAX_ALLPATH_SAMPLE = 1000
MAX_HOPS = 3


def rank_predictions(target_truth, all_true, predictions, max_rank) -> Tuple[int, bool]:
    rank = 1
    found_target = False
    for ind, (res, _) in enumerate(predictions):
        if res == target_truth:
            found_target = True
            break
        elif res in all_true:
            # filter, don't penalize for highly ranking a different true statements
            continue
        else:
            rank += 1
    if not found_target:
        rank = max_rank
    return rank

def naive_evaluation_single_sample(F: EvCBR, q, output,
                                   args, num_repeats:int = 1,
                                   s_cases:int = 10, s_cases_cov:int=5, s_paths:int = 25):
    max_rank = len(F.KG.all_nodes())
    while True:
        eval_payload = q.get()

        if eval_payload is None:
            break

        F.cycle_cache = dict() # purge cycle cache after each test

        s_p_o, truth_pred_objs = eval_payload

        eval_s, eval_p, eval_o = s_p_o

        fake_induction_triples = []
        for t in F.KG.triples((eval_s, None, None)):
            fake_induction_triples.append(t)
        reverse_combos = []

        ep_truths = defaultdict(lambda: set())
        for p,o in truth_pred_objs:
            ep_truths[p].add(o)

        forecast_properties = list(ep_truths.keys())

        # precompute similar cases to avoid recomputing for every property and repeat.
        # similar cases should be the same across all tests for a particular event.
        ##############################################################F.set_forecast_triples(dummy_triples=fake_induction_triples)
        precomp_sim_cases = F.collect_similar_events_with_target_relation(dummy_target_uri=eval_s,
                                                                          target_relation=eval_p)
        precomp_sim_cases_refined, pscs, case_connection_scores = F.refine_similar_cases_with_target_forecasts(
            target_entity=eval_s, candidates=precomp_sim_cases, forecast_properties=forecast_properties,
            forecast_step_relation=eval_p,
            print_info=False)
        precomp_similar_cases_final = []

        # select the "cause->effect" link where the effect's properties are most similar to our target forecast's relations.
        # if there are multiple "best" connections choose one at random.
        for ind, sim_case in enumerate(precomp_sim_cases_refined[:s_cases]):
            connected_effects = case_connection_scores[sim_case]
            max_effect_score = max(connected_effects.values())
            best_options = [k for k,v in connected_effects.items() if v==max_effect_score]
            if len(best_options) > 1:
                connection_choice = random.choice(best_options)
            else:
                connection_choice = best_options[0]
            precomp_similar_cases_final.append((sim_case, connection_choice))

        precomp_sim_cases_refined_EFF, pscs_EFF, case_connection_scores_EFF = F.refine_similar_cases_with_target_forecasts_effect_coverage(
            target_entity=eval_s, candidates=precomp_sim_cases, forecast_properties=forecast_properties,
            forecast_step_relation=eval_p,
            selected_cases=precomp_similar_cases_final,
            print_info=False)

        for ind, sim_case in enumerate(precomp_sim_cases_refined_EFF[:s_cases_cov]):
            connected_effects = case_connection_scores_EFF[sim_case]
            max_effect_score = max(connected_effects.values())
            best_options = [k for k,v in connected_effects.items() if v==max_effect_score]
            if len(best_options) > 1:
                connection_choice = random.choice(best_options)
            else:
                connection_choice = best_options[0]
            precomp_similar_cases_final.append((sim_case, connection_choice))

        no_fwd_preds = 0
        no_bkd_preds = 0
        total_preds = 0

        this_ranks = []
        for (target_rel, target_truth) in truth_pred_objs:
            this_property_ranks = {"prop_uri": target_rel, "prop_truth": target_truth,
                                   'inner_ranks': [],
                                   'inner_ranks_reversed': [],
                                   'rev_plus_fwd': []
                                   }
            this_ranks.append(this_property_ranks)

        # for rep_n in range(num_repeats):
        forecast_res = F.forecast_effects(
            triples_for_inductive_forecast=fake_induction_triples,
            dummy_target_uri=eval_s,
            forecast_relations=forecast_properties,
            max_hops=MAX_HOPS, sample_case_count=s_cases, sample_case_cov_count=s_cases_cov, sample_path_count=s_paths,
            print_info=False,
            precomputed_similar_cases=precomp_similar_cases_final,
            dummy_connecting_relation_uri=eval_p,
            prevent_inverse_paths=args.prevent_inverse_paths
        )

        for ep_ind, (target_rel, target_truth) in enumerate(truth_pred_objs):

            sorted_forecast_res = forecast_res.sorted_property_prediction(property_uri=target_rel)

            rank = rank_predictions(target_truth=target_truth, all_true=ep_truths[target_rel],
                                    predictions=sorted_forecast_res, max_rank=max_rank)

            this_ranks[ep_ind]["inner_ranks"].append(rank)

        ##################
        ##################
        ##################
        ##################
        if args.do_reverse_and_predict:
            reverse_res = F.forecast_effect_reverse_predictions(
                prop_forecasts=forecast_res.property_entity_support,
                dummy_target_uri=eval_s,
                triples_for_inductive_forecast=fake_induction_triples,
                similar_case_effects=forecast_res.similar_cause_effect_pairs,
                max_hops=MAX_HOPS,sample_path_count=s_paths,
                prevent_inverse_paths=args.prevent_inverse_paths
            )
            combine_prop_order = reverse_res.property_order
            reverse_prediction_support = reverse_res.property_prediction_support

            all_rev_in_top10 = True
            for ep_ind, (target_rel, target_truth) in enumerate(truth_pred_objs):
                p_truths = ep_truths[target_rel]
                if len(forecast_res.sorted_property_prediction(property_uri=target_rel)) == 0:
                    no_fwd_preds += 1
                if len(reverse_prediction_support[target_rel]) == 0:
                    no_bkd_preds += 1
                total_preds += 1

                rev_rank = rank_predictions(target_truth=target_truth, all_true=p_truths,
                                              predictions=reverse_prediction_support[target_rel], max_rank=max_rank)
                this_ranks[ep_ind]['inner_ranks_reversed'].append(rev_rank)
                if rev_rank > 10:
                    all_rev_in_top10 = False

                rpf_score = dict()
                fwd_sorted = {k:v for (k,v) in forecast_res.sorted_property_prediction(property_uri=target_rel)}
                bkd_sorted = {k:v for (k,v) in reverse_prediction_support[target_rel]}
                for k,v in fwd_sorted.items():
                    rpf_score[k] = v+bkd_sorted.get(k,0)
                sorted_rpf = sorted(rpf_score.items(), key=lambda x: x[1], reverse=True)
                rev_plus_fwd_rank = rank_predictions(target_truth=target_truth, all_true=p_truths,
                                              predictions=sorted_rpf, max_rank=max_rank)
                this_ranks[ep_ind]['rev_plus_fwd'].append(rev_plus_fwd_rank)

            ################
            reverse_input_support = reverse_res.property_input_support

            input_props = defaultdict(lambda: set())
            for t in fake_induction_triples:
                input_props[t[1]].add(t[2])
            ppc = {k:len(v) for k,v in ep_truths.items()}
            ############## COMBINER LOGIC REMOVED
            combiner_res = []
            reverse_combos.append(combiner_res)

        if not args.do_reverse_and_predict:
            output[(eval_s,eval_p,eval_o)] = {"cause_uri": eval_s, "effect_uri": eval_o, "forecast_property_ranks": this_ranks}
        else:
            output[(eval_s,eval_p,eval_o)] = {"cause_uri": eval_s, "effect_uri": eval_o,
                              "forecast_property_ranks": this_ranks,
                              "property_combos": reverse_combos,
                                              }

    print("worker finished")
    return

def enhance_kg_connections_to_superclasses(kg, expansion_rels: List[URIRef]):
    def recursive_expand_targets(kg, expansion_rels, next_ents, seen_ents):
        new_next_ents = set()
        for e in next_ents:
            for r in expansion_rels:
                for o in kg.objects(subject=e, predicate=r):
                    new_next_ents.add(o)
        new_next_ents = new_next_ents - next_ents - seen_ents
        seen_ents = seen_ents.union(new_next_ents)
        if new_next_ents:
            recursive_expand_targets(kg, expansion_rels, new_next_ents, seen_ents)

        return seen_ents

    new_triples = set()
    for n in kg.all_nodes():
        n_po = list(kg.predicate_objects(subject=n))

        sp_n = list(kg.subject_predicates(object=n))

        superclasses = recursive_expand_targets(kg, expansion_rels, {n}, set())
        for sup_c in superclasses:
            for s,p in sp_n:
                new_triples.add((s,p,sup_c))

    for nt in new_triples:
        kg.add(nt)

def main(args):

    pp_dir = (DATA_DIR / args.evcbr_pp_data_dir).resolve()
    testdata_dir = (DATA_DIR / args.pp_data_dir).resolve()
    save_dir = (DATA_DIR / args.save_dir).resolve()
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # %%
    main_kg = Graph()

    # load the graph based on training triples
    with open((testdata_dir / "train.txt").resolve(), "r") as f:
        for line in f:
            t = line.strip().split("\t")
            main_kg.add((URIRef(t[0]), URIRef(t[1]), URIRef(t[2])))
    NX_KG = rdflib_to_networkx_graph(main_kg)

    with open(Path.resolve(pp_dir / "subclass_matrix.pkl"), "rb") as f:
        subclass_matrix = pickle.load(f)
    with open(Path.resolve(pp_dir / "subclass_idf.pkl"), "rb") as f:
        subclass_idf = pickle.load(f)
    with open(Path.resolve(pp_dir / "ent_rel_matrix.pkl"), "rb") as f:
        ent_rel_matrix = pickle.load(f)
    with open(Path.resolve(pp_dir / "e2i.pkl"), "rb") as f:
        e2i = pickle.load(f)
    with open(Path.resolve(pp_dir / "r2i.pkl"), "rb") as f:
        r2i = pickle.load(f)
    pc = {}
    pc["NX_KG"] = NX_KG
    pc["subclass_matrix"] = subclass_matrix
    pc["subclass_idf"] = subclass_idf
    pc['ent_rel_matrix'] = ent_rel_matrix
    pc['e2i'] = e2i
    pc['r2i'] = r2i

    ## expand the kg AFTER the networkx graph has been set up
    # enhance_kg_connections_to_superclasses(main_kg, [WDT_SUBCLASSOF])

    print("finished setting up")

    # load ground truth test data
    test_truth = defaultdict(lambda: [])
    with open((testdata_dir / "test.txt").resolve(), "r") as f:
        for line in f:
            t = line.strip().split("\t")
            s, p, o = URIRef(t[0]), URIRef(t[1]), URIRef(t[2])
            test_truth[s].append((p, o))

    hp = 0
    np = 0
    h3pc = 0
    n3pc = 0
    eval_options = []
    bad_tests = 0
    AN = set(main_kg.all_nodes())
    with open((testdata_dir / "test_connections.txt"), "r") as f:
        for line in f:
            t = line.strip().split("\t")
            s, p, o = URIRef(t[0]), URIRef(t[1]), URIRef(t[2])
            bad_test = False
            for true_p, true_o in test_truth[o]:
                if true_o not in AN:
                    bad_tests += 1
                    bad_test = True
                    break
            if bad_test:
                continue
            eval_options.append(((s,p,o), test_truth[o]))
            for pp,oo in test_truth[o]:
                if nx.has_path(NX_KG, s, oo):
                    hp += 1
                    if nx.shortest_path_length(NX_KG, s, oo) <= 4:
                        h3pc += 1
                    else:
                        n3pc += 1
                else:
                    np += 1
                    n3pc += 1
    print(f"{bad_tests} bad test cases removed")
    print(f"has path: {hp}, no path: {np}")
    print(f"has 3hop path: {h3pc}, no 3hop: {n3pc}")

    print(f"{len(eval_options)} connections to perform evaluation.")

    n_rep = 1
    n_case = args.n_cases
    n_case_cov = args.n_cases_coverage
    n_paths = args.n_paths
    progress = 0

    q = mp.Queue(maxsize=args.processes)
    m = mp.Manager()
    output_dict = m.dict()
    output_dict["num repeats"] = n_rep
    output_dict["num similar cases to sample"] = n_case
    output_dict["num similar cases coverage"] = args.n_cases_coverage
    output_dict["num paths to sample"] = n_paths
    output_dict["no inv"] = args.prevent_inverse_paths
    output_dict["data dir"] = args.evcbr_pp_data_dir

    # Uncomment the following code to recover from previous runs, specifying the right reload_target
    # reload_target = "eval_res_fullcombo_progress_4946.pkl"
    # with open((save_dir / reload_target).resolve(), "rb") as f:
    #     print(f"loaded previous evals")
    #     prev_res = pickle.load(f)
    # for k,v in prev_res.items():
    #     # the following line was to pick out entries that mistakenly had no combos computed for them
    #     # newer progress files should be loading up everything, since the missing combos should be fixed.
    #     # in newer progress files, instances where no combos are present can occur if there just weren't any predictions
    #     # made for the target entity
    #     if isinstance(v, dict):# and len(v.get('property_combos', [[]])[0]) > 1:
    #         output_dict[k] = v
    #     else:
    #         print(k,v)
    # prev_res = None

    F = EvCBR(preload_content=pc, KG=main_kg, preprocessed_data_dir=pp_dir)
    pool = mp.Pool(args.processes, initializer=naive_evaluation_single_sample, initargs=(
        F, q, output_dict,
        args,
        n_rep, n_case, n_case_cov, n_paths
    ))

    print(f"starting, skipping {len(output_dict)-3} entries already completed")
    finished_ind = 0
    starttime = time.time()
    for payload in eval_options:
        finished_ind += 1
        # if a payload's first part (the triple) is already in the dict, it has been completed by a previous run.
        if payload[0] in output_dict.keys():
            continue

        q.put(payload)

        progress += 1
        if (progress < 100 and progress % 10 == 0) or (progress >= 100 and progress % 100 == 0):
            print(f"progress: {finished_ind} triples finished - {(finished_ind)/len(eval_options)}. {time.time()-starttime} seconds elapsed.")
        if progress > 1 and progress % 500 == 0:
            with open((save_dir / f"eval_res_fullcombo_progress_{finished_ind}.pkl").resolve(), "wb") as f:
                pickle.dump(dict(output_dict), f)
            print(f"saved at {finished_ind} evals")
    for _ in range(args.processes):
        q.put(None)
    pool.close()
    pool.join()
    print("finished all")
    print(f"progress: {progress/len(eval_options)}. {time.time()-starttime} seconds elapsed. saving")
    with open((save_dir / "eval_res.pkl").resolve(), "wb") as f:
        pickle.dump(dict(output_dict), f)
    print("finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp_data_dir", type=str, default="evcbr_pp_wn18rr")
    parser.add_argument("--evcbr_pp_data_dir", type=str, default="pp_wn18rr")
    parser.add_argument("--save_dir", type=str, default="eval_res_wn18rr")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--n_cases", type=int, default=5)
    parser.add_argument("--n_cases_coverage", type=int, default=5)
    parser.add_argument("--n_paths", type=int, default=20)
    parser.add_argument("--do_reverse_and_predict", action='store_true', default=False)
    parser.add_argument("--longer_combo_test", action='store_true', default=False)
    parser.add_argument("--do_preprocess_branching_paths", action='store_true', default=False)
    parser.add_argument("--prevent_inverse_paths", action='store_true', default=False)
    args = parser.parse_args()
    main(args)

