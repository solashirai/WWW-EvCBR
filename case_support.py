from dataclasses import dataclass
from typing import Dict, Set, Tuple, List
from rdflib import URIRef
from similar_cause_effect_choices import SimilarCauseEffectChoices


@dataclass
class CaseSupport:
    property_entity_support: Dict[URIRef, Dict[URIRef, float]]
    similar_cause_effect_pairs: List[SimilarCauseEffectChoices]
    c_to_e_paths: Set[Tuple[Tuple[URIRef, str]]]

    def sorted_property_prediction(self, *, property_uri: URIRef) -> List[Tuple[URIRef, float]]:
        prop_dict = self.property_entity_support[property_uri]
        return sorted(prop_dict.items(), key=lambda x: x[1], reverse=True)
