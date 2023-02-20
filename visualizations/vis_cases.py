import io
import pydotplus
from IPython.display import display, Image
from visualizations.modified_rdf2dot import mrdf2dot
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS
from similar_cause_effect_choice import SimilarCauseEffectChoice
from typing import List
from utils import *
import random


def visualize(g: Graph) -> None:
    stream = io.StringIO()
    mrdf2dot(g, stream, opts = {display})
    dg = pydotplus.graph_from_dot_data(stream.getvalue())
    dg.set_size('"15,8!"')

    # displaying image seemed to sometimes run into memory errors, so just save the image as a
    # png then have jupyter display it separately
    png = dg.create_png()
    display(Image(png))


def make_case_graph(c: SimilarCauseEffectChoice, kg: Graph, connecting_prop: URIRef,
                    cause_props: List[URIRef], effect_props: List[URIRef],
                    workaround_country: URIRef = None) -> Graph:
    case_graph = Graph()
    case_graph.bind("wd", WD)
    case_graph.bind("wdt", WDT)

    # the property connecting the "cause" and "effect" isn't included in the data, so supply it manually
    case_graph.add((c.cause, connecting_prop, c.effect))

    # only visualize properties that are relevant to our prediction's input/output since otherwise there will
    # often be way too many properties to display nicely.
    # for country, some entries have a LOT of countries entered. for such cases we will manually work around them
    for cp in cause_props:
        if cp==WDT["P17"] and len(c.cause_properties.get(cp, [])) > 3: # entry has many matches to country
            # country usually is a cause for a match when there are many entries, so use a manually supplied country
            # to match so that we are showing somewhat relevant info
            if workaround_country in c.cause_properties[cp]:
                case_graph.add((c.cause, cp, workaround_country))
                other_countries = random.choices(c.cause_properties[cp], k=3)
                for oc in other_countries:
                    case_graph.add((c.cause, cp, oc))

            # add in this dummy node to conform with the visualization's formatting
            DummyNode = URIRef("http://example.com/omitted")
            DummyLabel = Literal(f"{len(c.cause_properties.get(cp, []))-3} additional connections omitted")
            case_graph.add((c.cause, cp, DummyNode))
            case_graph.add((DummyNode, RDFS.label, DummyLabel))
        else:
            for o in c.cause_properties.get(cp, []):
                case_graph.add((c.cause, cp, o))
    for ep in effect_props:
        for o in c.effect_properties.get(ep, []):
            case_graph.add((c.effect, ep, o))

    # we're assuming that kg includes label info for these nodes
    for n in case_graph.all_nodes():
        label = kg.value(subject=n, predicate=RDFS.label)
        if label:
            case_graph.add((n, RDFS.label, label))

    return case_graph


def visualize_supporting_case(c: SimilarCauseEffectChoice, kg: Graph, connecting_prop:URIRef,
                              cause_props: List[URIRef], effect_props: List[URIRef],
                              workaround_country: URIRef = None) -> None:
    cg = make_case_graph(c, kg, connecting_prop, cause_props, effect_props, workaround_country)
    visualize(cg)
