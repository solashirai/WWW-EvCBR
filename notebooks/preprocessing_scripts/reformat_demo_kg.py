#%%
from evcbr import EvCBR
from utils import *
from rdflib import Graph, Literal

#%%

kg_file = (DATA_DIR /"wikidata_cc_full_3_hop.ttl").resolve()
full_kg_file = (DATA_DIR / "demo_kg"/"wikidata_cc_full_3_hop.nt").resolve()
model_kg_file = (DATA_DIR / "demo_kg"/"wikidata_cc_nolit_3_hop.nt").resolve()
preprocessed_sim_dir = (DATA_DIR / "pp_wiki_full").resolve()
full_kg = Graph()
full_kg.parse(str(kg_file), format='ttl')
# remove literals form the kg to use in the model
model_kg = Graph()
for (s,p,o) in full_kg:
    if not isinstance(o, Literal):
        model_kg.add((s,p,o))

#%%
full_kg.serialize(str(full_kg_file), format="nt")
model_kg.serialize(str(model_kg_file), format="nt")

