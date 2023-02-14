from experiments.preprocess_data_for_evcbr import vectorize_graph
from utils import *
from rdflib import Graph, Literal

wiki_input_dir = (DATA_DIR / "wiki_full").resolve()
wiki_output_dir = (DATA_DIR / "pp_wiki_full").resolve()
if not os.path.isdir(wiki_input_dir):
    os.mkdir(wiki_input_dir)
if not os.path.isdir(wiki_output_dir):
    os.mkdir(wiki_output_dir)

# dump the whole dataset into a "train" text file
print("dumping graph to txt")
g = Graph()
data_file = (DATA_DIR / "demo_kg"/"wikidata_cc_nolit_3_hop.nt").resolve()
g.parse(str(data_file), format='nt')
with open((wiki_input_dir / "train.txt").resolve(), 'w', encoding='utf-8') as f:
    for t in g:
        f.write(f"{t[0]}\t{t[1]}\t{t[2]}\n")

entities = dict()
for e in g.all_nodes():
    if isinstance(e, Literal):
        continue
    entities[e] = len(entities)
relations = dict()
all_rels = set(g.predicates(subject=None, object=None))
for r in all_rels:
    relations[r] = len(relations)
rev_entities = {v: k for k, v in entities.items()}
rev_relations = {v: k for k, v in relations.items()}
with open((wiki_input_dir / "entities.dict").resolve(), "w", encoding='utf-8') as f:
    for i in range(len(rev_entities)):
        f.write(f"{i}\t{rev_entities[i]}\n")
with open((wiki_input_dir / "relations.dict").resolve(), "w", encoding='utf-8') as f:
    for i in range(len(rev_relations)):
        f.write(f"{i}\t{rev_relations[i]}\n")
print("finished.")

print("starting preprocessing")
# run vectorization preprocessing
vectorize_graph(in_dir=wiki_input_dir, out_dir=wiki_output_dir, do_outgoing_edges=True, do_two_hop=False,
                              do_superclasses=[WDT_SUBCLASSOF, WDT_INSTANCEOF],
                vectorize_rdflib_graph=g)
print("finished")
