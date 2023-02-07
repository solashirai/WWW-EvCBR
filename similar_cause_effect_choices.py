from dataclasses import dataclass
from typing import Dict, List, Tuple
from rdflib import URIRef


@dataclass
class SimilarCauseEffectChoices:
    cause: URIRef
    effect: URIRef
    cause_properties: Dict[URIRef, List[URIRef]]
    effect_properties: Dict[URIRef, List[URIRef]]
