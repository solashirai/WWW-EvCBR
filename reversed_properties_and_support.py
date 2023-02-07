from dataclasses import dataclass
from typing import List, Dict, Tuple
from rdflib import URIRef


@dataclass
class ReversedPropertiesAndSupport:
    property_order: List[URIRef]
    property_prediction_support: Dict[URIRef, Dict[URIRef, float]]
    property_max_input_scores: Dict[URIRef, Dict[URIRef, float]]
    property_input_support: Dict[URIRef, Dict[URIRef, float]]

