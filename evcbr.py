from utils import *
import networkx as nx
from collections import defaultdict
from rdflib import URIRef
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
from typing import List, Dict, Tuple, Set
import numpy as np
import random
import pickle
from scipy import sparse
from scipy.sparse import dok_matrix
import time
from pathlib import Path
from case_support import CaseSupport
from similar_cause_effect_choices import SimilarCauseEffectChoices
from reversed_properties_and_support import ReversedPropertiesAndSupport


SMOOTHING_CONSTANT = 5

class EvCBR:

    def __init__(self, *, KG, preprocessed_data_dir, preload_content=None):
        if preload_content:
            self.subclass_matrix = preload_content["subclass_matrix"]
            self.subclass_idf = preload_content["subclass_idf"]
            self.ent_rel_matrix = preload_content["ent_rel_matrix"]
            self.e2i = preload_content["e2i"]
            self.i2e = {v:k for k,v in self.e2i.items()}
            self.r2i = preload_content["r2i"]
            self.NX_KG = preload_content["NX_KG"]
            self.KG = KG
        else:
            self.KG = KG
            self.NX_KG = rdflib_to_networkx_graph(self.KG)
            with open(Path.resolve(preprocessed_data_dir / "subclass_matrix.pkl"), "rb") as f:
                self.subclass_matrix = pickle.load(f)
            with open(Path.resolve(preprocessed_data_dir / "subclass_idf.pkl"), "rb") as f:
                self.subclass_idf = pickle.load(f)
            with open(Path.resolve(preprocessed_data_dir / "ent_rel_matrix.pkl"), "rb") as f:
                self.ent_rel_matrix = pickle.load(f)
            with open(Path.resolve(preprocessed_data_dir / "e2i.pkl"), "rb") as f:
                self.e2i = pickle.load(f)
                self.i2e = {v:k for k,v in self.e2i.items()}
            with open(Path.resolve(preprocessed_data_dir / "r2i.pkl"), "rb") as f:
                self.r2i = pickle.load(f)
                self.i2r = {v:k for k,v in self.r2i.items()}

        self.normalized_subclass_idf = self.subclass_idf / (np.sum(self.subclass_idf))
        self.idf_weighted_subclass_matrix = self.subclass_matrix.multiply(self.normalized_subclass_idf).tocsr()

        self.total_triples = len(self.KG)

        self.dummy_triples = []

        # precomputing the KG into a dict of dicts is faster, because rdflib just isn't very fast
        self.triple_dict_forward = dict()
        self.triple_dict_backward = dict()

        for s in self.KG.all_nodes():
            pred_dict = defaultdict(lambda: set())
            for p,o in self.KG.predicate_objects(subject=s):
                pred_dict[p].add(o)
            self.triple_dict_forward[s] = dict(pred_dict)
        for o in self.KG.all_nodes():
            pred_dict_rev = defaultdict(lambda: set())
            for s,p in self.KG.subject_predicates(object=o):
                pred_dict_rev[p].add(s)
            self.triple_dict_backward[o] = dict(pred_dict_rev)

        self.cycle_cache = dict()

    def set_forecast_triples(self, dummy_triples):
        self.added_dummy_triples = []
        for t in dummy_triples:
            if t not in self.KG:
                self.KG.add(t)
                self.added_dummy_triples.append(t)

                if t[0] not in self.triple_dict_forward.keys():
                    self.triple_dict_forward[t[0]] = {}
                if t[1] not in self.triple_dict_forward[t[0]].keys():
                    self.triple_dict_forward[t[0]][t[1]] = set()
                self.triple_dict_forward[t[0]][t[1]].add(t[2])

                if t[2] not in self.triple_dict_backward.keys():
                    self.triple_dict_backward[t[2]] = {}
                if t[1] not in self.triple_dict_backward[t[2]].keys():
                    self.triple_dict_backward[t[2]][t[1]] = set()
                self.triple_dict_backward[t[2]][t[1]].add(t[0])


    def clean_forecast_triples(self):
        for t in self.added_dummy_triples:
            self.KG.remove(t)
            self.triple_dict_forward[t[0]][t[1]].remove(t[2])
            self.triple_dict_backward[t[2]][t[1]].remove(t[0])
        self.added_dummy_triples = []

    def get_outgoing_paths_to_targets(self, *, source_node, start_nodes,
                                      target_nodes: Set[URIRef], target_invalid_node,
                                      max_hops):
        all_paths = []
        for start_node in start_nodes:
            temp_target_nodes = target_nodes
            if start_node in target_nodes:
                all_paths.append([(source_node, start_node)])
                temp_target_nodes = target_nodes - {start_node}
            for path in nx.all_simple_edge_paths(self.NX_KG, start_node, temp_target_nodes, cutoff=max_hops-1):
                # don't traverse through source_node
                goodpath = True
                for step in path:
                    if source_node in step or target_invalid_node in step:
                        goodpath = False
                if goodpath:
                    all_paths.append([(source_node, start_node)]+path)

        return all_paths

    def fast_get_and_format_cycles(self, *, target_uri, target_connectors, target_invalid_node, max_hops):
        if target_uri in self.cycle_cache:
            return self.cycle_cache[target_uri]

        cycles = []

        for connecting_ent in target_connectors:
            if connecting_ent == target_uri:
                continue
            for path in nx.all_simple_edge_paths(self.NX_KG, connecting_ent, target_uri, cutoff=max_hops-1):
                goodpath = True
                for step in path:
                    if target_invalid_node in step:
                        goodpath = False
                if goodpath:
                    cycles.append([(target_uri, connecting_ent)]+path)
        self.cycle_cache[target_uri] = cycles
        return cycles

    def follow_rdf_path_fastish(self, subj: URIRef, remaining_path: Tuple[Tuple[URIRef, str]],
                        path_nodes: List[URIRef],
                        path_cache: Dict,
                        ignore_nodes: Set[URIRef] = set(),
                        strict_test_ignore_incoming_to: Set[URIRef] = set(),
                        max_next_step: int = 1000) -> Dict[URIRef, int]:
        return self.follow_rdf_path(subj=subj, remaining_path=remaining_path, path_nodes=path_nodes,
                                    path_cache=path_cache, ignore_nodes=ignore_nodes,
                                    strict_test_ignore_incoming_to=strict_test_ignore_incoming_to,
                                    # strict_test_ignore_incoming_to=set(),
                                    max_next_step=max_next_step)

    def follow_rdf_path(self, subj: URIRef, remaining_path: Tuple[Tuple[URIRef, str]],
                        path_nodes: List[URIRef],
                        path_cache: Dict,
                        ignore_nodes: Set[URIRef] = set(),
                        strict_test_ignore_incoming_to: Set[URIRef] = set(),
                        max_next_step: int = 1000) -> Dict[URIRef, int]:

        if not remaining_path:
            # return [(subj, path_nodes)]
            return {subj: 1}
        pred = remaining_path[0][0]
        dir = remaining_path[0][1]
        path_result = defaultdict(lambda: 0)
        if dir == "forward":
            next_nodes = self.triple_dict_forward[subj].get(pred, set()) - {subj} - ignore_nodes - strict_test_ignore_incoming_to
        else:
            if subj in strict_test_ignore_incoming_to:
                return {}
            next_nodes = self.triple_dict_backward[subj].get(pred, set()) - {subj} - ignore_nodes

        if len(next_nodes) > max_next_step:
            next_nodes = random.sample(next_nodes, k=max_next_step)

        # we want to get all next_nodes to be output, but expect many repeats.
        # so we will cache results that have already been computed and add them to the path results.
        for o in next_nodes:
            cache_tuple = (o, remaining_path[1:])
            if cache_tuple in path_cache.keys():
                # path_result += path_cache[cache_tuple]
                for k,v in path_cache[cache_tuple].items():
                    path_result[k] += v
            else:
                updated_path = path_nodes + [o]
                path_cache[cache_tuple] = self.follow_rdf_path(subj=o, remaining_path=remaining_path[1:],
                                                               path_nodes=updated_path,
                                                               strict_test_ignore_incoming_to=strict_test_ignore_incoming_to,
                                                               ignore_nodes=ignore_nodes,
                                                               max_next_step=max_next_step, path_cache=path_cache)

                for k,v in path_cache[cache_tuple].items():
                    path_result[k] += v

        return dict(path_result)

    def forecast_effects(self, *,
                         triples_for_inductive_forecast: List,
                         dummy_target_uri: URIRef,
                         forecast_relations: List[URIRef],
                         max_hops: int = 3, sample_case_count: int = 5, sample_case_cov_count: int = 5,
                         sample_path_count: int = 25,
                         print_info: bool = False,
                         precomputed_similar_cases: List[Tuple[URIRef, URIRef]] = None,
                         prevent_inverse_paths: bool = False,
                         dummy_connecting_relation_uri: URIRef = WDT_HASEFFECT) -> \
            CaseSupport:
        forecast_relations = set(forecast_relations)
        total_case_count = sample_case_count + sample_case_cov_count
        # self.set_forecast_triples(dummy_triples=triples_for_inductive_forecast)

        inductive_input_relations = set()
        for t in triples_for_inductive_forecast:
            if t[0] == dummy_target_uri:
                inductive_input_relations.add(t[1])

        # do an ugly combination of querying for stuff in rdf then doing path stuff in networkx
        # eventually need to remove literals
        if print_info:
            print("starting alg")
        if precomputed_similar_cases is None:
            sim_cases = self.collect_similar_events_with_target_relation(dummy_target_uri=dummy_target_uri,
                                                                         target_relation=dummy_connecting_relation_uri)
            sim_cases_refined, pscs, case_connection_scores = self.refine_similar_cases_with_target_forecasts(
                target_entity=dummy_target_uri, candidates=sim_cases,
                forecast_properties=forecast_relations,
                forecast_step_relation=dummy_connecting_relation_uri,
                print_info=False)
            similar_cases_final = []

            # select the "cause->effect" link where the effect's properties are most similar to our target forecast's relations.
            # if there are multiple "best" connections choose one at random.
            for ind, sim_case in enumerate(sim_cases_refined[:sample_case_count]):
                connected_effects = case_connection_scores[sim_case]
                max_effect_score = max(connected_effects.values())
                best_options = [k for k, v in connected_effects.items() if v == max_effect_score]
                if len(best_options) > 1:
                    connection_choice = random.choice(best_options)
                else:
                    connection_choice = best_options[0]
                similar_cases_final.append((sim_case, connection_choice))

            sim_cases_refined_EFF, pscs_EFF, case_connection_scores_EFF = self.refine_similar_cases_with_target_forecasts_effect_coverage(
                target_entity=dummy_target_uri, candidates=sim_cases,
                forecast_properties=forecast_relations,
                forecast_step_relation=dummy_connecting_relation_uri,
                selected_cases=similar_cases_final,
                print_info=False)

            for ind, sim_case in enumerate(sim_cases_refined_EFF[:sample_case_cov_count]):
                connected_effects = case_connection_scores_EFF[sim_case]
                max_effect_score = max(connected_effects.values())
                best_options = [k for k, v in connected_effects.items() if v == max_effect_score]
                if len(best_options) > 1:
                    connection_choice = random.choice(best_options)
                else:
                    connection_choice = best_options[0]
                similar_cases_final.append((sim_case, connection_choice))

            similar_cases_final = similar_cases_final[:total_case_count]

        else:
            similar_cases_refined = precomputed_similar_cases
            similar_cases_final = similar_cases_refined[:total_case_count]

        similar_cause_effects = similar_cases_final
        original_sc_count = len(similar_cause_effects)
        if len(similar_cause_effects) > total_case_count:
            similar_cause_effects = random.sample(similar_cause_effects, total_case_count)

        if print_info:
            print(f"{original_sc_count} similar cases retrieved.")
            print(f"Sampling {total_case_count} similar cases for up to {sample_path_count} paths")


        sim_ce = []
        for spe in similar_cause_effects:
            cause = spe[0]
            effect = spe[1]
            cause_props = defaultdict(lambda: [])
            for p,o in self.KG.predicate_objects(subject=cause):
                cause_props[p].append(o)
            effect_props = defaultdict(lambda: [])
            for p,o in self.KG.predicate_objects(subject=effect):
                effect_props[p].append(o)
            sim_ce.append(SimilarCauseEffectChoices(cause=cause, effect=effect,
                                                    cause_properties=dict(cause_props),
                                                    effect_properties=dict(effect_props)))


        starttime = time.time()
        prop_path_stats = {p:
            {
                'path_counts': defaultdict(lambda: 0),
                'path_precision': defaultdict(lambda: []),
                'path_successful_results': defaultdict(lambda: 0),
                'path_total_results': defaultdict(lambda: 0),
                'total_sampled_paths': 0
            }
            for p in forecast_relations
        }

        findpath_times = 0
        follow_times = 0
        all_seen_paths = set()
        for sc in sim_ce:
            res_cache = dict()
            sc_effect_properties = sc.effect_properties
            sc_effect_uri = sc.effect
            sc_cause_uri = sc.cause
            relevant_cause_firststep_vals = set()
            for relevant_prop in inductive_input_relations:
                for o in sc.cause_properties.get(relevant_prop, []):
                    relevant_cause_firststep_vals.add(o)


            get_cycles = False
            sc_effect_targets = set([ent for k,v in sc_effect_properties.items() for ent in v])
            # including the cause when performing the search for all paths leads to no paths being returned
            # so handle cycles separately
            if sc_cause_uri in sc_effect_targets:
                get_cycles = True
                sc_effect_targets.remove(sc_cause_uri)
            # get paths to all potential targets at once
            all_paths = self.get_outgoing_paths_to_targets(source_node=sc_cause_uri,
                                                           start_nodes=relevant_cause_firststep_vals,
                                                           target_nodes=sc_effect_targets,
                                                           target_invalid_node=sc_effect_uri,
                                                           max_hops=max_hops)

            if get_cycles:
                smallcycles = self.fast_get_and_format_cycles(target_uri=sc_cause_uri,
                                                              target_connectors=relevant_cause_firststep_vals,
                                                              target_invalid_node=sc_effect_uri,
                                                              max_hops=max_hops)
                all_paths += smallcycles

            # sort out the paths based on the relation they correspond to
            prop_paths = {k:[] for k in sc_effect_properties.keys() if k in forecast_relations}
            for prop in prop_paths.keys():
                for p in all_paths:
                    # paths are a list of tuples (from, to), so check the last connection's outgoing edge to check if
                    # it matches with an entity for the target property
                    if p[-1][-1] in sc_effect_properties[prop]:
                        prop_paths[prop].append(p)

            for prop, paths in prop_paths.items():

                if len(paths) > sample_path_count:
                    paths = random.sample(paths, sample_path_count)

                all_path_edges = []
                for path in paths:
                    branching_path_edges = []
                    prev_node = sc_cause_uri
                    has_good_start = False
                    for ind, e in enumerate(path):
                        edge_triple_all = self.NX_KG.get_edge_data(e[0], e[1])["triples"]
                        next_paths = []
                        next_prev_node = None
                        for edge_triple in edge_triple_all:
                            if edge_triple[0] != prev_node:
                                edge_dir = "backward"
                                next_prev_node = edge_triple[0]
                            else:
                                edge_dir = "forward"
                                next_prev_node = edge_triple[2]

                            e_label = edge_triple[1]
                            if ind == 0 and not has_good_start: # check on the first step whether any of the paths will be good
                                if edge_dir == "forward" and e_label in inductive_input_relations:
                                    has_good_start = True
                            next_paths.append((e_label, edge_dir))

                        if not has_good_start: # don't continue on for bad paths
                            break

                        prev_node = next_prev_node
                        new_branching_path_edges = []
                        if not branching_path_edges:
                            for npath in next_paths:
                                new_branching_path_edges.append([npath])
                        else:
                            for npath in next_paths:
                                for bp in branching_path_edges:
                                    new_branching_path_edges.append(bp+[npath])
                        branching_path_edges = new_branching_path_edges

                    if not has_good_start: # if paths don't have a good starting point, continue on
                        continue

                    for bpe in branching_path_edges:
                        # only use paths where the first step is relevant to our input
                        if (bpe[0][1] == "forward" and bpe[0][0] in inductive_input_relations):
                            # only use paths where there isn't an inverse at any step
                            bad_bpe = False
                            for bpe_ind in range(1, len(bpe)):
                                prev_rel_dir = bpe[bpe_ind-1]
                                this_rel_dir = bpe[bpe_ind]
                                if prevent_inverse_paths:
                                    if prev_rel_dir[0] == this_rel_dir[0]: # same relation
                                        if prev_rel_dir[1] != this_rel_dir[1]: # opposite direction
                                            # i.e. the inverse path is being used.
                                            # this should still allow cases like symmetric relations since they should have
                                            # the same direction
                                            bad_bpe = True
                            if not bad_bpe:
                                all_path_edges.append(bpe)

                if len(all_path_edges) > sample_path_count:
                    all_path_edges = random.sample(all_path_edges, sample_path_count)
                all_path_edges = [tuple(ape) for ape in all_path_edges]
                prop_path_stats[prop]['total_sampled_paths'] += len(all_path_edges)

                miniprop_counts = defaultdict(lambda: 0)
                miniprop_successes = defaultdict(lambda: 0)
                miniprop_totals = defaultdict(lambda: 0)
                for path_edges in all_path_edges:
                    miniprop_counts[path_edges] += 1
                    ft = time.time()
                    if (sc_cause_uri, tuple(path_edges)) in res_cache.keys():
                        path_forecasts = res_cache[(sc_cause_uri, tuple(path_edges))]
                    else:
                        path_forecasts = self.follow_rdf_path_fastish(subj=sc_cause_uri, remaining_path=tuple(path_edges),
                                                              strict_test_ignore_incoming_to=set([dummy_target_uri]),
                                                              path_nodes=[], path_cache=dict())
                        res_cache[(sc_cause_uri, tuple(path_edges))] = path_forecasts
                    follow_times += time.time()-ft
                    if not path_forecasts:
                        # if no forecast is reached by the path, continue.
                        continue
                    # fast ver
                    for eff_ent in sc_effect_properties[prop]:
                        if eff_ent in path_forecasts.keys():
                            miniprop_successes[path_edges] += path_forecasts[eff_ent]
                    miniprop_totals[path_edges] += sum(path_forecasts.values())

                # new stuff
                for path_edges in set(all_path_edges):
                    if miniprop_totals[path_edges] > 0:
                        prop_path_stats[prop]['path_counts'][path_edges] += miniprop_counts[path_edges]
                        prop_path_stats[prop]['path_successful_results'][path_edges] += miniprop_successes[path_edges]
                        prop_path_stats[prop]['path_total_results'][path_edges] += miniprop_totals[path_edges]
                        prop_path_stats[prop]['path_precision'][path_edges].append(
                            miniprop_successes[path_edges]/(miniprop_totals[path_edges]+SMOOTHING_CONSTANT))

        for p, ps in prop_path_stats.items():
            if ps['total_sampled_paths'] == 0:
                continue
            for k in ps['path_successful_results'].keys():
                ps['path_precision'][k] = (ps['path_successful_results'][k]/(ps['path_total_results'][k]+SMOOTHING_CONSTANT))

        if print_info:
            print(f"time taken: {time.time() - starttime}")
            print(f"time taken for nx pathfinding: {findpath_times}")
            print(f"time taken for following paths and computing precision: {follow_times}")

            print("following paths to find results")
        starttime = time.time()

        prop_forecasts = dict()
        for p, ps in prop_path_stats.items():
            inner_starttime = time.time()
            valid_path_count = 0
            total_paths = 0
            forecast_support = defaultdict(lambda: 0.0)

            best_paths = sorted(ps['path_precision'].items(), key=lambda x: x[1], reverse=True)

            for path, precision in best_paths:

                path_forecast = self.follow_rdf_path_fastish(subj=dummy_target_uri, remaining_path=tuple(path),
                                                      strict_test_ignore_incoming_to=set([dummy_target_uri]),
                                                      path_nodes=[], path_cache=dict())
                if path_forecast:
                    valid_path_count += 1
                for forecast_ent, forecast_ent_count in path_forecast.items():
                    total_paths += 1
                    path_val = precision
                    forecast_support[forecast_ent] += path_val*forecast_ent_count

            for k, v in forecast_support.items():
                forecast_support[k] = v / total_paths

            if print_info:
                print(f"property {p} - {valid_path_count} valid paths followed to identify {len(forecast_support)} entities")

                print(f"time taken: {time.time() - inner_starttime}")
            prop_forecasts[p] = dict(forecast_support)

        if print_info:
            print(f"time taken for all properties: {time.time()-starttime}")

        forecast_res = CaseSupport(property_entity_support=prop_forecasts,
                                   similar_cause_effect_pairs=sim_ce,
                                   c_to_e_paths=all_seen_paths)

        return forecast_res

    def get_and_refine_similar_cases(self, *, target_entity: URIRef) -> List:
        similar_cases = self.collect_similar_events_with_effects(target_entity)
        similar_cases_refined, scs = self.refine_similar_cases(target_entity=target_entity,
                                                               candidates=similar_cases)
        return similar_cases_refined

    def refine_similar_cases(self, *, target_entity: URIRef, candidates: List[URIRef], print_info: bool = False):

        # precompute the cosine similarity of outgoing connections for the target entity, then reuse those
        # to get the similarity of each candidate.
        target_outgoing = defaultdict(lambda: set())
        for p,o in self.KG.predicate_objects(subject=target_entity):
            target_outgoing[p].add(o)

        sim_weight = dict()
        total_weights = 0
        for p in target_outgoing.keys():
            max_pred_weight = 0
            obj_weight = dict()
            for o in target_outgoing[p]:
                obj_occ = len(list(self.KG.subject_predicates(object=o)))
                match_count = len(list(self.KG.subjects(predicate=p, object=o)))
                # set weight as the log of P(relation p given object o) / P(triples incoming to o in the KG)
                # similar to pointwise mutual information...?
                weight = np.log10((match_count/obj_occ) / (obj_occ/self.total_triples))
                obj_weight[o] = weight
                max_pred_weight = max(max_pred_weight, weight)
            # each predicate weight differs based on the outgoing node it matches to.
            # this is only relevant if there are multiple outgoing relations for the forecast.
            sim_weight[p] = obj_weight
            total_weights += max_pred_weight
        # apply some 'normalization' to the weights to try to avoid any of them being too influential.
        # divide everything by the sum of the maximum weight for each predicate, so that a perfect match would end up
        # with a similarity of 1 in theory. if multiple triples for the same relation exist, the most influential one is
        # used for this normalization
        sim_weight = {k:{kk:vv/total_weights for kk,vv in v.items()} for k,v in sim_weight.items()}

        if print_info:
            print("weights")
            print(sim_weight)

        target_outgoing_sims = defaultdict(lambda: dict())
        out_to_id = dict()
        sparse_choices = []
        for k,v in target_outgoing.items():
            for ent in v:
                if ent not in out_to_id:
                    out_to_id[ent] = len(out_to_id.keys())
                    sparse_choices.append(self.e2i[ent])

        weighted_sim = self.idf_class_sim(self.idf_weighted_subclass_matrix[sparse_choices], self.idf_weighted_subclass_matrix).T

        for k, v in target_outgoing.items():
            for ent in v:
                target_outgoing_sims[k][ent] = weighted_sim[out_to_id[ent]]

        candidate_scores = dict()
        candidate_score_breakdown = defaultdict(lambda: [])
        for c in candidates:
            candidate_outgoing = defaultdict(lambda: set())
            for p,o in self.KG.predicate_objects(subject=c):
                candidate_outgoing[p].add(o)
            total_score = 0
            for k, sims in target_outgoing_sims.items():
                max_sim = 0
                for cand_out_ent in candidate_outgoing[k]:
                    if cand_out_ent in self.e2i.keys():
                        for ent, sim in sims.items():
                            max_sim = max(max_sim, sim_weight[k][ent]*sim[0, self.e2i[cand_out_ent]])
                total_score += max_sim
                candidate_score_breakdown[c].append(max_sim)
            candidate_scores[c] = total_score

        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        sc_scores = [candidate_score_breakdown[tup[0]] for tup in sorted_candidates]
        sorted_candidates = [tup[0] for tup in sorted_candidates]
        return sorted_candidates, sc_scores

    def refine_similar_cases_with_target_forecasts(
            self, *, target_entity: URIRef, candidates: List[URIRef],
            forecast_properties: List[URIRef], forecast_step_relation: URIRef,
            print_info: bool = False):
        forecast_properties = set(forecast_properties)

        # precompute the cosine similarity of outgoing connections for the target entity, then reuse those
        # to get the similarity of each candidate.
        target_outgoing = defaultdict(lambda: set())
        for p,o in self.KG.predicate_objects(subject=target_entity):
            target_outgoing[p].add(o)

        sim_weight = dict()
        total_weights = 0
        for p in target_outgoing.keys():
            max_pred_weight = 0
            obj_weight = dict()
            for o in target_outgoing[p]:
                obj_occ = len(list(self.KG.subject_predicates(object=o)))
                match_count = len(list(self.KG.subjects(predicate=p, object=o)))
                # set weight as the log of P(relation p given object o) / P(triples incoming to o in the KG)
                weight = np.log10((match_count/obj_occ) / (obj_occ/self.total_triples))
                obj_weight[o] = weight
                max_pred_weight = max(max_pred_weight, weight)
            # each predicate weight differs based on the outgoing node it matches to.
            # this is only relevant if there are multiple outgoing relations for the forecast.
            sim_weight[p] = obj_weight
            total_weights += max_pred_weight
        # apply some 'normalization' to the weights to try to avoid any of them being too influential.
        # divide everything by the sum of the maximum weight for each predicate, so that a perfect match would end up
        # with a similarity of 1 in theory. if multiple triples for the same relation exist, the most influential one is
        # used for this normalization
        sim_weight = {k:{kk:vv/total_weights for kk,vv in v.items()} for k,v in sim_weight.items()}

        if print_info:
            print("weights")
            print(sim_weight)

        target_outgoing_sims = defaultdict(lambda: dict())
        out_to_id = dict()
        sparse_choices = []
        for k,v in target_outgoing.items():
            for ent in v:
                if ent not in out_to_id:
                    out_to_id[ent] = len(out_to_id.keys())
                    sparse_choices.append(self.e2i[ent])

        weighted_sim = self.idf_class_sim(self.idf_weighted_subclass_matrix[sparse_choices], self.idf_weighted_subclass_matrix).T

        for k, v in target_outgoing.items():
            for ent in v:
                target_outgoing_sims[k][ent] = weighted_sim[out_to_id[ent]]

        candidate_scores = dict()
        case_connection_scores = dict()
        candidate_score_breakdown = defaultdict(lambda: [])
        for c in candidates:
            cand_conn_dict = dict()
            candidate_outgoing = defaultdict(lambda: set())
            for p,o in self.KG.predicate_objects(subject=c):
                candidate_outgoing[p].add(o)
            total_score = 0
            for k, sims in target_outgoing_sims.items():
                max_sim = 0
                for cand_out_ent in candidate_outgoing[k]:
                    if cand_out_ent in self.e2i.keys():
                        for ent, sim in sims.items():
                            max_sim = max(max_sim, sim_weight[k][ent]*sim[0, self.e2i[cand_out_ent]])
                total_score += max_sim
                candidate_score_breakdown[c].append(max_sim)
            best_outgoing_sim = 0
            for outgoing_forecast_ent in self.KG.objects(subject=c, predicate=forecast_step_relation):
                outgoing_relations = set(self.triple_dict_forward[outgoing_forecast_ent].keys())
                # get jaccard sim
                this_outgoing_sim = len(outgoing_relations.intersection(forecast_properties)) / len(outgoing_relations.union(forecast_properties))
                cand_conn_dict[outgoing_forecast_ent] = this_outgoing_sim
                if this_outgoing_sim > best_outgoing_sim:
                    best_outgoing_sim = this_outgoing_sim

            candidate_scores[c] = total_score*best_outgoing_sim
            case_connection_scores[c] = cand_conn_dict

        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        sc_scores = [candidate_score_breakdown[tup[0]] for tup in sorted_candidates]
        sorted_candidates = [tup[0] for tup in sorted_candidates]
        return sorted_candidates, sc_scores, case_connection_scores

    def refine_similar_cases_with_target_forecasts_effect_coverage(
            self, *, target_entity: URIRef, candidates: List[URIRef],
            forecast_properties: List[URIRef], forecast_step_relation: URIRef,
            selected_cases: List[Tuple[URIRef, URIRef]] = None,
            print_info: bool = False):
        forecast_properties = set(forecast_properties)

        # precompute the cosine similarity of outgoing connections for the target entity, then reuse those
        # to get the similarity of each candidate.
        target_outgoing = defaultdict(lambda: set())
        for p,o in self.KG.predicate_objects(subject=target_entity):
            target_outgoing[p].add(o)

        sim_weight = dict()
        total_weights = 0
        for p in target_outgoing.keys():
            max_pred_weight = 0
            obj_weight = dict()
            for o in target_outgoing[p]:
                obj_occ = len(list(self.KG.subject_predicates(object=o)))
                match_count = len(list(self.KG.subjects(predicate=p, object=o)))
                # set weight as the log of P(relation p given object o) / P(triples incoming to o in the KG)
                weight = np.log10((match_count/obj_occ) / (obj_occ/self.total_triples))
                obj_weight[o] = weight
                max_pred_weight = max(max_pred_weight, weight)
            # each predicate weight differs based on the outgoing node it matches to.
            # this is only relevant if there are multiple outgoing relations for the forecast.
            sim_weight[p] = obj_weight
            total_weights += max_pred_weight
        # apply some 'normalization' to the weights to try to avoid any of them being too influential.
        # divide everything by the sum of the maximum weight for each predicate, so that a perfect match would end up
        # with a similarity of 1 in theory. if multiple triples for the same relation exist, the most influential one is
        # used for this normalization
        sim_weight = {k:{kk:vv/total_weights for kk,vv in v.items()} for k,v in sim_weight.items()}

        if print_info:
            print("weights")
            print(sim_weight)

        target_outgoing_sims = defaultdict(lambda: dict())
        out_to_id = dict()
        sparse_choices = []
        for k,v in target_outgoing.items():
            for ent in v:
                if ent not in out_to_id:
                    out_to_id[ent] = len(out_to_id.keys())
                    sparse_choices.append(self.e2i[ent])

        weighted_sim = self.idf_class_sim(self.idf_weighted_subclass_matrix[sparse_choices], self.idf_weighted_subclass_matrix).T

        for k, v in target_outgoing.items():
            for ent in v:
                target_outgoing_sims[k][ent] = weighted_sim[out_to_id[ent]]

        candidate_scores = dict()
        case_connection_scores = dict()
        candidate_score_breakdown = defaultdict(lambda: [])

        previously_selected_set = set(selected_cases) if selected_cases else set()
        for c in candidates:
            cand_conn_dict = dict()
            candidate_outgoing = defaultdict(lambda: set())
            for p,o in self.KG.predicate_objects(subject=c):
                candidate_outgoing[p].add(o)
            total_score = 0
            for k, sims in target_outgoing_sims.items():
                max_sim = 0
                for cand_out_ent in candidate_outgoing[k]:
                    if cand_out_ent in self.e2i.keys():
                        for ent, sim in sims.items():
                            max_sim = max(max_sim, sim_weight[k][ent]*sim[0, self.e2i[cand_out_ent]])
                total_score += max_sim
                candidate_score_breakdown[c].append(max_sim)
            best_outgoing_sim = 0
            for outgoing_forecast_ent in self.KG.objects(subject=c, predicate=forecast_step_relation):
                if (c, outgoing_forecast_ent) in previously_selected_set:
                    continue
                outgoing_relations = set(self.triple_dict_forward[outgoing_forecast_ent].keys())
                # get jaccard sim
                this_outgoing_sim = len(outgoing_relations.intersection(forecast_properties)) / len(forecast_properties)
                cand_conn_dict[outgoing_forecast_ent] = this_outgoing_sim
                if this_outgoing_sim > best_outgoing_sim:
                    best_outgoing_sim = this_outgoing_sim

            candidate_scores[c] = (1+total_score)*best_outgoing_sim
            case_connection_scores[c] = cand_conn_dict

        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        sc_scores = [candidate_score_breakdown[tup[0]] for tup in sorted_candidates]
        sorted_candidates = [tup[0] for tup in sorted_candidates]
        return sorted_candidates, sc_scores, case_connection_scores



    def idf_class_sim(self, u, v):
        '''
        based on jaccard sim, with weighting applied.
        assumes u is smaller than v, since the transpose operation will then be faster.
        '''
        top = v.dot(u.T)
        bot = np.repeat(v.sum(axis=1), top.shape[1], axis=1)+np.repeat(u.sum(axis=1), top.shape[0], axis=1).T-top
        return top / bot

    def collect_similar_effects(self, relation_list: List[URIRef], initial_candidate_limit: int = None) -> List[URIRef]:

        dummy_target_vec = np.zeros((1, len(self.r2i.keys())))
        for r in relation_list:
            dummy_target_vec[0, self.r2i[r]] = 1

        relation_overlap = sparse.csr_matrix.dot(self.effect_event_er_matrix, dummy_target_vec.T).T
        top_candidates = np.argsort(relation_overlap)[0][::-1]

        if initial_candidate_limit is None:
            return [self.i2e[c] for c in top_candidates if relation_overlap[0, c] != 0]
        else:
            candidates = []
            current_cand_ind = 0
            current_top_score = 0
            current_top_tie = []
            while len(candidates) < initial_candidate_limit or current_cand_ind >= len(top_candidates):
                current_candidate = top_candidates[current_cand_ind]
                if relation_overlap[current_candidate] > current_top_score:
                    current_top_score = relation_overlap[current_candidate]
                    if current_top_tie:
                        if len(candidates) + len(current_top_tie) < initial_candidate_limit:
                            for ctt in current_top_tie:
                                candidates.append(ctt)
                            current_top_tie = []
                        else:
                            candidates += random.sample(current_top_tie, initial_candidate_limit-len(candidates))
                            break
                elif relation_overlap[current_candidate] == current_top_score:
                    current_top_tie.append(current_candidate)
                else:
                    if current_top_tie:
                        if len(candidates) + len(current_top_tie) < initial_candidate_limit:
                            for ctt in current_top_tie:
                                candidates.append(ctt)
                        else:
                            candidates += random.sample(current_top_tie, initial_candidate_limit-len(candidates))
                    break
                current_cand_ind += 1

            return [self.i2e[c] for c in top_candidates]

    def collect_similar_events_with_target_relation(self, dummy_target_uri: URIRef, target_relation: URIRef, initial_candidate_limit: int = None) -> \
            List[URIRef]:
        '''
        For a given target entity, return a sorted list of the most similar entities present in the
        knowledge graph. The returned list contains the URI as well as the similarity score assigned
        to it, sorted in descending order.

        :param dummy_target_uri:
        :param initial_candidate_limit:
        :return:
        '''
        dummy_target_relations = set(self.KG.predicates(subject=dummy_target_uri, object=None))

        dummy_target_vec = np.zeros((1, len(self.r2i.keys())))
        for r in dummy_target_relations:
            dummy_target_vec[0, self.r2i[r]] = 1

        target_er_choices = dok_matrix((1, len(self.e2i.keys())))
        for subj, obj in self.KG.subject_objects(predicate=target_relation):
            if subj == dummy_target_uri: # don't count the forecast entity as "similar" to itself
                continue
            target_er_choices[0, self.e2i[subj]] = 1
        target_er_choices = target_er_choices.tocsr()
        target_relation_er_matrix = target_er_choices.T.multiply(self.ent_rel_matrix)

        relation_overlap = sparse.csr_matrix.dot(target_relation_er_matrix, dummy_target_vec.T).T
        top_candidates = np.argsort(relation_overlap)[0][::-1]
        if initial_candidate_limit:
            top_candidates = top_candidates[:initial_candidate_limit]

        return [self.i2e[c] for c in top_candidates if relation_overlap[0, c] != 0]

    def forecast_effect_reverse_predictions(
            self, *,
            prop_forecasts: Dict[URIRef, Dict[URIRef, float]],
            dummy_target_uri: URIRef,
            triples_for_inductive_forecast: List[Tuple[URIRef, URIRef, URIRef]],
            similar_case_effects: List[SimilarCauseEffectChoices],
            max_hops: int = 3, sample_path_count: int = 25,
            prevent_inverse_paths: bool = False
    ) -> ReversedPropertiesAndSupport:

        input_prop_targets = defaultdict(lambda: set())
        input_target_ents = set()
        for t in triples_for_inductive_forecast:
            if t[0] == dummy_target_uri:
                input_prop_targets[t[1]].add(t[2])
                input_target_ents.add(t[2])

        combine_prop_order = []
        forecast_property_set = set(prop_forecasts.keys())
        for p in prop_forecasts.keys():
            combine_prop_order.append(p)

        ##########
        prop_path_stats = {p:
            {
                'path_counts': defaultdict(lambda: 0),
                'path_precision': defaultdict(lambda: []),
                'path_successful_results': defaultdict(lambda: 0),
                'path_total_results': defaultdict(lambda: 0)
            }
            for p in input_prop_targets.keys()
        }

        findpath_times = 0
        follow_times = 0
        for sc in similar_case_effects:
            res_cache = dict()
            cause = sc.cause
            effect = sc.effect
            cause_prop_vals = {k: set(v) for k, v in sc.cause_properties.items() if k in input_prop_targets.keys()}
            relevant_effect_vals = set()
            for relevant_prop in combine_prop_order:
                for relevant_val in sc.effect_properties.get(relevant_prop, []):
                    relevant_effect_vals.add(relevant_val)

            get_cycles = False
            cause_vals = set([val for vals in cause_prop_vals.values() for val in vals])
            if effect in cause_vals:
                get_cycles = True
                cause_vals.remove(effect)
            # get paths to all potential targets at once

            all_paths = self.get_outgoing_paths_to_targets(source_node=effect,
                                                           start_nodes=relevant_effect_vals,
                                                           target_nodes=cause_vals,
                                                           target_invalid_node=cause,
                                                           max_hops=max_hops)

            if get_cycles:
                smallcycles = self.fast_get_and_format_cycles(target_uri=effect,
                                                target_connectors=relevant_effect_vals,
                                                target_invalid_node=cause,
                                                max_hops=max_hops)
                all_paths += smallcycles

            # sort out the paths based on the relation they correspond to
            prop_paths = {k: [] for k in cause_prop_vals.keys()}
            for prop in prop_paths.keys():
                for p in all_paths:
                    # paths are a list of tuples (from, to), so check the last connection's outgoing edge to check if
                    # it matches with an entity for the target property
                    if p[-1][-1] in cause_prop_vals[prop]:
                        prop_paths[prop].append(p)

            for prop, paths in prop_paths.items():

                if len(paths) > sample_path_count:
                    paths = random.sample(paths, sample_path_count)

                all_path_edges = []
                for path in paths:
                    branching_path_edges = []
                    prev_node = effect
                    has_good_start = False

                    for ind, e in enumerate(path):
                        edge_triple_all = self.NX_KG.get_edge_data(e[0], e[1])["triples"]
                        next_paths = []
                        for edge_triple in edge_triple_all:
                            if edge_triple[0] != prev_node:
                                edge_dir = "backward"
                                next_prev_node = edge_triple[0]
                            else:
                                edge_dir = "forward"
                                next_prev_node = edge_triple[2]

                            e_label = edge_triple[1]
                            next_paths.append((e_label, edge_dir))

                            if ind == 0 and not has_good_start: # check on the first step whether any of the paths will be good
                                if edge_dir == "forward" and e_label in forecast_property_set:
                                    has_good_start = True

                        if not has_good_start: # don't continue on for bad paths
                            break

                        prev_node = next_prev_node
                        new_branching_path_edges = []
                        if not branching_path_edges:
                            for npath in next_paths:
                                new_branching_path_edges.append([npath])
                        else:
                            for npath in next_paths:
                                for bp in branching_path_edges:
                                    new_branching_path_edges.append(bp + [npath])
                        branching_path_edges = new_branching_path_edges

                    if not has_good_start: # if paths don't have a good starting point, continue on
                        continue

                    # only use paths where the first step is relevant to our input
                    for bpe in branching_path_edges:
                        if bpe[0][1] == "forward" and bpe[0][0] in forecast_property_set:
                            bad_bpe = False
                            for bpe_ind in range(1, len(bpe)):
                                prev_rel_dir = bpe[bpe_ind - 1]
                                this_rel_dir = bpe[bpe_ind]
                                if prevent_inverse_paths:
                                    if prev_rel_dir[0] == this_rel_dir[0]:  # same relation
                                        if prev_rel_dir[1] != this_rel_dir[1]:  # opposite direction
                                            # i.e. the inverse path is being used.
                                            # this should still allow cases like symmetric relations since they should have
                                            # the same direction
                                            bad_bpe = True
                            if not bad_bpe:
                                all_path_edges.append(bpe)

                            all_path_edges.append(bpe)

                all_path_counts = defaultdict(lambda: 0)
                total_all_path_counts = len(all_path_edges)
                for ape in all_path_edges:
                    ape_key = tuple(ape)
                    all_path_counts[ape_key] += 1

                if len(all_path_edges) > sample_path_count:
                    all_path_edges = random.sample(all_path_edges, sample_path_count)
                all_path_edges = [tuple(ape) for ape in all_path_edges]

                miniprop_counts = defaultdict(lambda: 0)
                miniprop_successes = defaultdict(lambda: 0)
                miniprop_totals = defaultdict(lambda: 0)
                for path_edges in all_path_edges:
                    miniprop_counts[path_edges] += 1
                    ft = time.time()
                    if (effect, tuple(path_edges)) in res_cache.keys():
                        path_forecasts = res_cache[(effect, tuple(path_edges))]
                    else:
                        path_forecasts = self.follow_rdf_path_fastish(subj=effect, remaining_path=tuple(path_edges),
                                                              strict_test_ignore_incoming_to=set([dummy_target_uri]),
                                                              path_nodes=[], path_cache=dict())
                        res_cache[(effect, tuple(path_edges))] = path_forecasts
                    follow_times += time.time() - ft
                    if not path_forecasts:
                        # if no forecast is reached by the path, continue.
                        continue

                    for cause_ent in cause_prop_vals[prop]:
                        if cause_ent in path_forecasts.keys():
                            miniprop_successes[path_edges] += path_forecasts[cause_ent]
                    miniprop_totals[path_edges] += sum(path_forecasts.values())
                # new stuff
                for path_edges in set(all_path_edges):
                    if miniprop_totals[path_edges] > 0:
                        prop_path_stats[prop]['path_counts'][path_edges] += miniprop_counts[path_edges]
                        prop_path_stats[prop]['path_successful_results'][path_edges] += miniprop_successes[path_edges]
                        prop_path_stats[prop]['path_total_results'][path_edges] += miniprop_totals[path_edges]
                        prop_path_stats[prop]['path_precision'][path_edges].append(
                             miniprop_successes[path_edges]/(miniprop_totals[path_edges]+SMOOTHING_CONSTANT))

        for p, ps in prop_path_stats.items():
            for k in ps['path_successful_results'].keys():
                ps['path_precision'][k] = np.mean(ps['path_precision'][k])
        stime = time.time()

        startingpoints = {}
        for p, preds in prop_forecasts.items():
            startingpoints[p] = set(preds.keys())
        startpoint_scores = defaultdict(
            lambda:
            {p: {fent: 0 for fent_set in input_prop_targets.values() for fent in fent_set}
             for p in prop_forecasts.keys()}
        )

        res_cache = dict()
        for fprop, fent_set in input_prop_targets.items():
            ps = prop_path_stats[fprop]

            for path, count in ps['path_counts'].items():
                if len(path) > 1:
                    step0_pred = path[0][0]
                    step0_dir = path[0][1]
                    if step0_dir == "forward":
                        for startent in startingpoints.get(step0_pred, []):

                            # original version
                            if (startent, tuple(path[1:])) in res_cache.keys():
                                testmypath_forecast = res_cache[(startent, tuple(path[1:]))]
                            else:
                                testmypath_forecast = self.follow_rdf_path_fastish(subj=startent, remaining_path=path[1:],
                                                                      strict_test_ignore_incoming_to=set([dummy_target_uri]),
                                                                      path_nodes=[], path_cache=dict())
                                res_cache[(startent, tuple(path[1:]))] = testmypath_forecast

                            testmypath_forecast_set = set(testmypath_forecast.keys())
                            total_forecast_count = sum(testmypath_forecast.values())
                            path_val = ps['path_precision'][path]
                            for fent in fent_set:
                                if fent in testmypath_forecast_set:
                                    startpoint_scores[startent][step0_pred][fent] += (testmypath_forecast[fent]/total_forecast_count)*path_val

                else:
                    # currently assume only one property is being predicted, so if path is
                    # length 1 then just check if the entity is equal
                    path_val = ps['path_precision'][path]
                    for startent in startingpoints.get(path[0][0], []):
                        for fent in fent_set:
                            if fent == startent:
                                startpoint_scores[startent][path[0][0]][fent] += (1/len(startingpoints[path[0][0]]))*path_val

        combo_simplified4 = dict()
        backwards_effect_prop_support = dict()
        max_input_supports = dict()
        for cprop in combine_prop_order:
            max_prop_scores = {input_prop: 0 for input_prop in input_prop_targets.keys()}
            ent_to_prop = {}
            for prop, inent_set in input_prop_targets.items():
                for inent in inent_set:
                    ent_to_prop[inent] = prop
            ent_score_dict_v4 = {}
            for starter in startpoint_scores.keys():
                starter_score = sum(startpoint_scores[starter][cprop].values())
                if starter_score > 0:
                    starter_score_v4 = startpoint_scores[starter][cprop]
                    for inprop, inent_set in input_prop_targets.items():
                        for inent in inent_set:
                            if starter_score_v4[inent] > max_prop_scores[inprop]:
                                max_prop_scores[inprop] = starter_score_v4[inent]
                    ent_score_dict_v4[starter] = starter_score_v4

            ent_score_dict_v4_unsquashed = dict()

            for k, v in ent_score_dict_v4.items():
                ent_score_dict_v4_unsquashed[k] = {
                    vk: vv / max_prop_scores[ent_to_prop[vk]] if max_prop_scores[ent_to_prop[vk]] > 0 else 0 for vk, vv
                    in v.items()}
                support_vals = [vv / max_prop_scores[ent_to_prop[vk]] if max_prop_scores[ent_to_prop[vk]] > 0 else 0 for
                                vk, vv in v.items()]
                ent_score_dict_v4[k] = (max(support_vals)+ np.mean(support_vals)) * prop_forecasts[cprop][k]

            sorted_starters4 = sorted(ent_score_dict_v4.items(), key=lambda x: x[1], reverse=True)
            combo_simplified4[cprop] = sorted_starters4

            backwards_effect_prop_support[cprop] = ent_score_dict_v4_unsquashed

            max_input_supports[cprop] = None

        ##########
        res = ReversedPropertiesAndSupport(
            property_order=combine_prop_order,
            property_prediction_support=combo_simplified4,
            property_max_input_scores=max_input_supports,
            property_input_support=backwards_effect_prop_support
        )
        return res

