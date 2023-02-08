# EvCBR
This repository contains code for our WWW'23 paper "Event Prediction using Case-Based Reasoning over
Knowledge Graphs" (paper link to be added)

EvCBR is a case-based reasoning model for performing event prediction using knowledge graphs.
The main idea of our work is to (1) frame the task of event prediction as a 2-hop inductive link prediction task,
starting from the cause event and making predictions about the effect event's properties, and 
(2) retrieve cases of similar cause-effect event pairs in the KG in order to learn reasoning paths
that can be used to make predictions. Framing our task in this way allows us to make predictions about
new, unseen effect events only based on input properties about a cause event.

![EvCBR Overview](images/MotivExample1.pdf)

This repository also contains a copy of our KG of causal event triples, curated from Wikidata. 
The dataset is also hosted [here](https://zenodo.org/record/7196049#.Y0jVi9fMKUk).

### Preliminaries

We recommend that you create a new python virtual env and install the requirements. Our 
experiments were performed using Python 3.8

`python -m venv venv/`

`source env/bin/activate` 

`pip install -r requirements.txt`

## Setup and Running the Experiments

The experiments can be run using 4 scripts to (1) split the Wikidata-based event data into
train/test splits, (2) precompute stats and vectors for entity similarity, (3) run the 
EvCBR model, then (4) show results.

### Split Data

The following assumes that you have the wikidata dataset placed in the `data/` directory.
It's small enough to be uploaded to github so it is already included, but it is also available
at the aforementioned link on Zenodo. Additionally, we have files to specify which classes we
consider to be "events", and a file containing the subclass hierarchy of those events.

To split the data, run:

If desired, you can modify the default arguments such as the output directories. 

`python experiments/split_wikidata_dataset.py`

This will produce two new folders in `data/` - `pp_wiki/`, containing txt files of the 
triples in the train/test datasets, and `evcbr_pp_wiki/`, which is empty at this step but 
will be populated next.

### Preprocess Data

Next, run the script to preprocess the data. The default arguments will point to the correct
data, but you can also specify the input/output directories. From the previous step, the `pp_wiki/`
folder will be the default input, and `evcbr_pp_wiki/` is the default output.

`python experiments/preprocess_data_for_evcbr.py --process_wiki`

### Run EvCBR Model

To run our model on the experiment data, use the following command.

```
python experiments/run_evcbr_test.py 
    --do_reverse_and_predict 
    --pp_data_dir pp_wiki 
    --evcbr_pp_data_dir evcbr_pp_wiki 
    --save_dir wiki_results 
    --processes 8
    --n_cases 5 
    --n_cases_coverage 3 
    --n_paths 80
```

`--do_reverse_and_predict` will run EvCBR's additional refinement step. Without this tag, only the basic forward 
predictions will take place. Note that this flag will make the runtime longer.

`--processes` is used to enable multiprocessing, which is highly recommended. You will likely need
to adjust the number of processes being run based on what makes sense for your CPU.
Note that increasing the number of processes will also require more memory. In our current Wikidata tests,
each process can require arounf 2GB of memory.

`--n_cases` and `--n_cases_coverage` define how many cases from the KG to retrieve in order to
discover prediction paths.

`--n_paths` determines how many prediction paths are sampled from each case.

In our experiments using 8 processes, this step takes roughly 30 minutes.

### Display Results

Lastly, the results can be output and saved using 

`python experiments/show_evcbr_eval_results.py --eval_res_dir wiki_results --data_dir pp_wiki`

### Additional Experiments

Split train/valid/test data for FB15k-237 and WN18RR are also included in `data/`.
The `preprocess_data_for_evcbr.py` can be called with the `--process_fb'/`--process_wn` flags, and then
the last 2 experiment scripts can be run.
Please keep in mind that the FB15k dataset contains many more triples to predict, the KG is also more
densely connected, which can lead to very long runtimes.

## Citation

`bibtex citation coming soon`
