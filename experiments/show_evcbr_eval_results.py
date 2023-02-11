import pickle
from rdflib import URIRef
import numpy as np
from collections import defaultdict
from utils import *
import argparse
import rdflib


def order_combo_val(combo_tuples):
    sorted_combos = sorted(combo_tuples, key=lambda x: x[1], reverse=True)
    return sorted_combos


def order_combo_sum(combo_tuples):
    sorted_combos = sorted(combo_tuples, key=lambda x: sum(v for v in x[1].values()), reverse=True)
    return sorted_combos

def order_combo_sqrsum(combo_tuples):
    sorted_combos = sorted(combo_tuples, key=lambda x: sum(v**2 for v in x[1].values()), reverse=True)
    return sorted_combos


def order_combo_avg(combo_tuples):
    # actually the results should be the same as sum anyways?
    sorted_combos = sorted(combo_tuples, key=lambda x: np.mean([v for v in x[1].values()]), reverse=True)
    return sorted_combos


def score_prec(combo_choices, true_choices):
    prec_div = len(true_choices.keys())
    prec_correct = 0
    for k,v in combo_choices.items():
        true_ents = true_choices[k]
        if v in true_ents:
            prec_correct += 1
    return prec_correct/prec_div


def score_full_accuracy(combo_choices, true_choices):
    prec_div = len([vv for k,v in true_choices.items() for vv in v])
    prec_correct = 0
    for k,v in combo_choices.items():
        for pred_v in v:
            if pred_v in true_choices[k]:
                prec_correct += 1
    return prec_correct/prec_div


def score_which_recall(combo_choices, true_choices):
    recall_dict = {k:0 for k in true_choices.keys()}
    for k,v in combo_choices.items():
        if v in true_choices[k]:
            recall_dict[k] = 1
    return recall_dict

def score_recall(combo_choices, true_choices):
    recall_dict = {k:0 for k in true_choices.keys()}
    for k,v in combo_choices.items():
        if v in true_choices[k]:
            recall_dict[k] = 1
    return sum(recall_dict.values())/len(recall_dict)

def main(load_file, eval_res_dir, has_reverse, data_dir, save_file):

    with open(load_file, 'rb') as f:
        data = pickle.load(f)

    rankmax = 0
    with open((DATA_DIR/data_dir/"entities.dict").resolve(), 'r') as f:
        for line in f:
            rankmax += 1

    total_h1 = []
    total_h10 = []
    total_h100 = []
    total_ranks = []
    prop_ranks = defaultdict(lambda: [])
    nonzero_ranks = []
    nonzero_prop_ranks = defaultdict(lambda: [])
    rev_event_mrrs = defaultdict(lambda: [])
    rev_event_h5 = defaultdict(lambda: [])
    rev_event_h10 = defaultdict(lambda: [])
    rev_event_h100 = defaultdict(lambda: [])
    rpf_event_mrrs = defaultdict(lambda: [])
    rpf_event_h5 = defaultdict(lambda: [])
    rpf_event_h10 = defaultdict(lambda: [])
    rpf_event_h100 = defaultdict(lambda: [])

    if has_reverse == 1:
        total_r_h1 = []
        total_r_h10 = []
        total_r_h100 = []
        total_r_ranks = []
        prop_r_ranks = defaultdict(lambda: [])
        nonzero_r_ranks = []
        nonzero_prop_r_ranks = defaultdict(lambda: [])

        total_rpf_h1 = []
        total_rpf_h10 = []
        total_rpf_h100 = []
        total_rpf_ranks = []
        prop_rpf_ranks = defaultdict(lambda: [])

    prop_keys = []
    for k in data.keys():
        if not isinstance(k, str):
            prop_keys.append(k)

    firsthop_avg_ranks = defaultdict(lambda: [])
    cause_effect_type_truths = defaultdict(lambda: [])

    filter_n_cases = 1
    types_with_n_cases = 0
    is_inverse_relation = 0
    total_props = 0

    for k in prop_keys:
        if len(data[k]) < filter_n_cases:
            continue
        else:
            types_with_n_cases += 1

        cause = data[k]
        cause_ranks = []
        cause_rev_ranks = []
        cause_rpf_ranks = []
        target_cause_types = [k]
        combo_parts = 0
        for p in cause['forecast_property_ranks']:
            if p['prop_truth'] == cause['cause_uri']:
                is_inverse_relation += 1
            total_props += 1

            prank = np.mean(p['inner_ranks'])
            firsthop_avg_ranks[k[1]].append(prank)
            cause_ranks.append(prank)

            if prank != rankmax:
                nonzero_ranks.append(prank)
                nonzero_prop_ranks[p['prop_uri']].append(prank)

            prop_ranks[p['prop_uri']].append(prank)
            total_h1.append(1 if prank==1 else 0)
            total_h10.append(1 if prank<=10 else 0)
            total_h100.append(1 if prank<=100 else 0)
            total_ranks.append(prank)

            for t in target_cause_types:
                # cause_effect_type_predictions[t].append(p['predictions'][0][:10])
                cause_effect_type_truths[t].append(p['prop_truth'])

            if has_reverse == 1:
                rprank = np.mean(p['inner_ranks_reversed'])
                cause_rev_ranks.append(rprank)
                if rprank != rankmax:
                    nonzero_r_ranks.append(rprank)
                    nonzero_prop_r_ranks[p['prop_uri']].append(rprank)
                prop_r_ranks[p['prop_uri']].append(rprank)
                total_r_h1.append(1 if rprank==1 else 0)
                total_r_h10.append(1 if rprank<=10 else 0)
                total_r_h100.append(1 if rprank<=100 else 0)
                total_r_ranks.append(rprank)

                rpfrank = np.mean(p['rev_plus_fwd'])
                cause_rpf_ranks.append(rpfrank)
                prop_rpf_ranks[p['prop_uri']].append(rpfrank)
                total_rpf_h1.append(1 if rpfrank==1 else 0)
                total_rpf_h10.append(1 if rpfrank<=10 else 0)
                total_rpf_h100.append(1 if rpfrank<=100 else 0)
                total_rpf_ranks.append(rpfrank)

        rev_event_mrrs[k].append(np.mean([1/r for r in cause_rev_ranks]))
        rev_event_h5[k].append(np.mean([1 if r<=5 else 0 for r in cause_rev_ranks]))
        rev_event_h10[k].append(np.mean([1 if r<=10 else 0 for r in cause_rev_ranks]))
        rev_event_h100[k].append(np.mean([1 if r<=100 else 0 for r in cause_rev_ranks]))
        rpf_event_mrrs[k].append(np.mean([1/r for r in cause_rpf_ranks]))
        rpf_event_h5[k].append(np.mean([1 if r<=5 else 0 for r in cause_rpf_ranks]))
        rpf_event_h10[k].append(np.mean([1 if r<=10 else 0 for r in cause_rpf_ranks]))
        rpf_event_h100[k].append(np.mean([1 if r<=100 else 0 for r in cause_rpf_ranks]))

    print(f"number of inverses: {is_inverse_relation}, out of {total_props}.")

    prop_counts = []
    pair_counts = []
    for k in prop_keys:
        pair_counts.append(len(data[k]))
        cause = data[k]
        prop_counts.append(len(cause['forecast_property_ranks']))

    pred_num_content = dict()
    nonzero_pred_num_content = dict()
    pred_num_r_content = dict()
    nonzero_pred_num_r_content = dict()
    pred_mrr_content = dict()

    # get performance for each property
    for p in prop_ranks.keys():
        ranks = prop_ranks[p]
        if len(ranks) == 0:
            pred_content = f"{p}: {len(ranks)} occurrences\n-------\n"
            pred_num_content[p] = (len(ranks), pred_content)
            continue
        mr = np.mean(ranks)
        mrr = np.mean([1 / r for r in ranks])
        h1 = np.mean([1 if r == 1 else 0 for r in ranks])
        h10 = np.mean([1 if r <= 10 else 0 for r in ranks])
        pred_content = f"{p}: {len(ranks)} occurrences\n----\n{mr}\t{mrr}\t{h1}\t{h10}\n----\n"
        pred_num_content[p] = (len(ranks), pred_content)

        ranks_nz = nonzero_prop_ranks[p]
        if len(ranks_nz) == 0:
            mr, mrr, h1, h10 = 0,0,0,0
        else:
            mr = np.mean(ranks_nz)
            mrr = np.mean([1 / r for r in ranks_nz])
            h1 = np.mean([1 if r == 1 else 0 for r in ranks_nz])
            h10 = np.mean([1 if r <= 10 else 0 for r in ranks_nz])
        nonzero_pred_content = f"{p}:nonzero predictions: {len(ranks_nz)} occurrences\n----\n{mr}\t{mrr}\t{h1}\t{h10}\n----\n"
        nonzero_pred_num_content[p] = (len(ranks_nz), nonzero_pred_content)

        if has_reverse == 1:
            ranks = prop_r_ranks[p]
            mr = np.mean(ranks)
            mrr = np.mean([1 / r for r in ranks])
            h1 = np.mean([1 if r == 1 else 0 for r in ranks])
            h10 = np.mean([1 if r <= 10 else 0 for r in ranks])
            pred_r_content = f"{p}:rev: {len(ranks)} occurrences\n----\n{mr}\t{mrr}\t{h1}\t{h10}\n----\n"
            pred_num_r_content[p] = (len(ranks), pred_r_content)

            ranks_nz = nonzero_prop_r_ranks[p]
            if len(ranks_nz) == 0:
                mr, mrr, h1, h10 = 0,0,0,0
            else:
                mr = np.mean(ranks_nz)
                mrr = np.mean([1 / r for r in ranks_nz])
                h1 = np.mean([1 if r == 1 else 0 for r in ranks_nz])
                h10 = np.mean([1 if r <= 10 else 0 for r in ranks_nz])
            nonzero_pred_r_content = f"{p}:rev:nonzero predictions: {len(ranks_nz)} occurrences\n----\n{mr}\t{mrr}\t{h1}\t{h10}\n----\n"
            nonzero_pred_num_r_content[p] = (len(ranks_nz), nonzero_pred_r_content)
        pred_mrr_content[p] = (mrr, pred_content)
    sorted_pnc = sorted(pred_num_content.items(), key=lambda x: x[1][0], reverse=True)


    print(f"Writing out results to {save_file}")
    with open(save_file, 'w') as f:
        f.write(f"data processed at {data['data dir']}\n")
        f.write(f"test using {data['num similar cases to sample']} similar cases, "
                f"{data['num similar cases coverage']} coverage samples, "
                f"sampling {data['num paths to sample']} paths from each case, "
                f"with prevent inverse paths={data['no inv']}\n")
        f.write(f"{types_with_n_cases} event types with more than {filter_n_cases} events in the dataset\n")
        f.write(f"{round(np.mean(pair_counts),4)} average cause-effect events for each cause event type\n")
        f.write(f"number of inverses: {is_inverse_relation}, out of {total_props}. \n")
        f.write(f"number of properties for each effect event - mean: {round(np.mean(prop_counts),4)}, med: {np.median(prop_counts)}\n")
        f.write("====RESULTS===\n")
        f.write(f"all predicates\n            \tMRR  \tH@1      \tH@10      \tH@100\n")

        f.write(f"basic res  :\t{round(np.mean([1/r for r in total_ranks]), 4)}"
                f"\th@1:{round(np.mean(total_h1),4)}\th@10:{round(np.mean(total_h10),4)}\t"
                f"h@100:{round(np.mean(total_h100),4)}\n")
        if has_reverse == 1:
            f.write(f"refined res:\t{round(np.mean([1/r for r in total_r_ranks]), 4)}"
                    f"\th@1:{round(np.mean(total_r_h1),4)}\th@10:{round(np.mean(total_r_h10),4)}\t"
                    f"h@100:{round(np.mean(total_r_h100),4)}\n")
            f.write(f"bas+ref res:\t{round(np.mean([1/r for r in total_rpf_ranks]), 4)}"
                    f"\th@1:{round(np.mean(total_rpf_h1),4)}\th@10:{round(np.mean(total_rpf_h10),4)}\t"
                    f"h@100:{round(np.mean(total_rpf_h100),4)}\n")

        f.write("\nOne limitation of the current method is that we frequently see no predictions being made for a "
                "given property because\n"
                "no good prediction paths being identified from similar cases.\n")
        f.write(f"MRR for non-empty predictions: {round(np.mean([1 / r for r in nonzero_ranks]), 4)} - "
                f"({len(nonzero_ranks)} predictions made out of {len(total_ranks)} test predictions)\n")
        if has_reverse == 1:
            f.write(
                f"MRR for non-empty predictions, using the refinement method: "
                f"{round(np.mean([1 / r for r in nonzero_r_ranks]), 4)}\n"
                f"({len(nonzero_r_ranks)} predictions made out of {len(total_ranks)} test predictions)\n")

        f.write("\n========\n")
        f.write("\n\npredicate breakdown (MR/MRR/H@1/H@10):\n")

        for ind, pnc in enumerate(sorted_pnc):
            f.write(pnc[1][1])
            f.write(nonzero_pred_num_content[pnc[0]][1])
            if has_reverse == 1:
                f.write(pred_num_r_content[pnc[0]][1])
                f.write(nonzero_pred_num_r_content[pnc[0]][1])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_res_dir", type=str, default="eval_res_wn18rr")
    parser.add_argument("--data_dir", type=str, default="pp_wn18rr")
    parser.add_argument("--save_file", type=str, default="eval_res_output.txt")
    args = parser.parse_args()
    data_dir = args.data_dir
    eval_res_dir = (DATA_DIR / args.eval_res_dir).resolve()
    load_file = (DATA_DIR / args.eval_res_dir / "eval_res.pkl").resolve()
    save_file = (eval_res_dir / args.save_file).resolve()
    main(load_file, eval_res_dir, True, data_dir, save_file)

