import io
import pydotplus
from IPython.display import display, Image
from visualizations.modified_rdf2dot import mrdf2dot
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS
from similar_cause_effect_choice import SimilarCauseEffectChoice
from typing import List, Dict, Set, Tuple
import networkx as nx
from utils import *
import random
from ipysigma import Sigma
from case_support import CaseSupport


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


def new_make_case_graph(c: SimilarCauseEffectChoice, connecting_prop: URIRef,
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
        if cp == WDT["P17"] and len(c.cause_properties.get(cp, [])) > 3:  # entry has many matches to country
            # country usually is a cause for a match when there are many entries, so use a manually supplied country
            # to match so that we are showing somewhat relevant info
            if workaround_country in c.cause_properties[cp]:
                case_graph.add((c.cause, cp, workaround_country))
                other_countries = random.choices(c.cause_properties[cp], k=3)
                for oc in other_countries:
                    case_graph.add((c.cause, cp, oc))

            # add in this dummy node to conform with the visualization's formatting
            DummyNode = URIRef("http://example.com/omitted")
            DummyLabel = Literal(f"{len(c.cause_properties.get(cp, [])) - 3} additional connections omitted")
            case_graph.add((c.cause, cp, DummyNode))
            case_graph.add((DummyNode, RDFS.label, DummyLabel))
        else:
            for o in c.cause_properties.get(cp, []):
                case_graph.add((c.cause, cp, o))
    for ep in effect_props:
        for o in c.effect_properties.get(ep, []):
            case_graph.add((c.effect, ep, o))

    return case_graph


def new_collective_make_case_graph(cs: List[SimilarCauseEffectChoice], kg: Graph, connecting_prop: URIRef,
                                   cause_props: List[URIRef], effect_props: List[URIRef],
                                   workaround_country: URIRef = None) -> Graph:
    case_graph = Graph()
    case_graph.bind("wd", WD)
    case_graph.bind("wdt", WDT)

    for c in cs:
        cg = new_make_case_graph(c, connecting_prop, cause_props, effect_props, workaround_country)
        case_graph += cg

    return case_graph


def setup_sigma_graph(kg, connecting_prop=WDT_HASEFFECT, path_connections=[],
                      custom_node_labels: Dict[URIRef, str] = dict(),
                      custom_edge_labels: Dict[Tuple[URIRef, URIRef], str] = dict(),
                      custom_node_sizes: Dict[URIRef, float] = dict()) -> Sigma:
    node_labels = collect_wikidata_labels(list(kg.all_nodes()))
    # any label defined in custom_node_labels will have priority
    # node_labels |= custom_node_labels # this op requires higher python version (3.10+?) for dicts
    for k, v in custom_node_labels.items():
        node_labels[k] = v
    property_labels = collect_wikidata_property_labels(list(set(kg.predicates(subject=None, object=None))))

    g = nx.DiGraph()

    edges = set()
    edge_labels = defaultdict(lambda: set())
    connecting_edges = []
    path_edges = []
    for (s, p, o) in kg:
        if p != RDFS_LABEL:
            edge_lab = property_labels.get(p.split("/")[-1], "")
            if (s, o) in custom_edge_labels:
                edge_lab = custom_edge_labels[(s, o)] + edge_lab
            edges.add((s, o))
            edge_labels[(s, o)].add(edge_lab)
            if p == connecting_prop:
                connecting_edges.append((s, o))
    # workaround to handle cases where multiple edges exist
    # multidigraph causes problems with ipysigma
    edge_labels = dict(edge_labels)
    edges = [(e[0], e[1], {"label": ', '.join(list(edge_labels.get(e, '')))}) for e in edges]

    g.add_edges_from(edges)
    for n in g.nodes:
        if n not in node_labels:
            node_labels[n] = ""

    node_sizes = dict(g.degree)
    for k, v in custom_node_sizes.items():
        node_sizes[k] += v

    edge_colors = {k: "property" for k in g.edges}
    node_colors = {n: "entity" for n in g.nodes}
    edge_sizes = {k: 1 for k in g.edges}

    for e in connecting_edges:
        edge_colors[e] = "connection"
        edge_sizes[e] = 2
        node_colors[e[0]] = "cause-effect"
        node_colors[e[1]] = "cause-effect"
    for ind, path_edges in enumerate(path_connections):
        for e in path_edges:
            edge_sizes[e] = 2
            edge_colors[e] = ind  # "blue"

    return Sigma(g, node_size=node_sizes, edge_size=edge_sizes,
                 label_density=2,  # show_all_labels=True,
                 default_edge_type="arrow", node_border_color_from="node",
                 node_label=node_labels,
                 node_color=node_colors, edge_color=edge_colors, )


def query_for_path(kg: Graph, path: List, start: URIRef, end: URIRef,
                   cause_triples: List[Tuple[URIRef, URIRef, URIRef]]) -> List[Tuple[URIRef, URIRef, URIRef]]:
    for t in cause_triples:
        kg.add(t)
    selection_str = ""
    for ind, (step_rel, step_dir) in enumerate(path):
        prev_node = start.n3() if ind == 0 else f"?n{ind}"
        next_node = f"?n{ind + 1}" if ind != len(path) - 1 else end.n3()
        step_str = f"{prev_node} {step_rel.n3()} {next_node}" if step_dir == "forward" else f"{next_node} {step_rel.n3()} {prev_node}"
        selection_str += step_str + " .\n"
    sparql_str = f"""
    CONSTRUCT {{{selection_str}}}
    WHERE {{
      {selection_str}
    }} LIMIT 1
    """
    res = kg.query(sparql_str)

    for t in cause_triples:
        kg.remove(t)

    return list(res)


def make_prediction_path_graph(kg: Graph, cause_triples: List[Tuple[URIRef, URIRef, URIRef]],
                               prediction_property: URIRef, prediction_ent: URIRef, prediction_paths,
                               pathcount: int = 5, connecting_prop:URIRef=WDT_HASEFFECT) -> Tuple[Graph, List[List]]:
    case_graph = Graph()
    case_graph.bind("wd", WD)
    case_graph.bind("wdt", WDT)

    for t in cause_triples:
        case_graph.add(t)
    cause_ent = cause_triples[0][0]
    effect_ent = WD["EFFECT_EVENT"]
    # the property connecting the "cause" and "effect" isn't included in the data, so supply it manually
    case_graph.add((cause_ent, connecting_prop, effect_ent))
    case_graph.add((effect_ent, prediction_property, prediction_ent))

    sorted_pathscores = sorted(prediction_paths.items(), key=lambda x: x[1], reverse=True)
    top_paths = [path_det for (path_det, path_score) in sorted_pathscores[:pathcount]]
    all_path_edges = []
    for p in top_paths:
        path_edges = []
        path_trips = query_for_path(kg, p, cause_ent, prediction_ent, cause_triples)
        for t in path_trips:
            case_graph.add(t)
            path_edges.append((t[0], t[2]))
        all_path_edges.append(path_edges)
    return case_graph, all_path_edges


def visualize_prediction_path(kg: Graph, cause_triples: List[Tuple[URIRef, URIRef, URIRef]],
                              prediction_property: URIRef, prediction_ent: URIRef,
                              prediction_paths,
                              pathcount: int = 5) -> Sigma:
    cg, pe = make_prediction_path_graph(kg, cause_triples, prediction_property, prediction_ent, prediction_paths,
                                        pathcount)
    cause_ent = cause_triples[0][0]
    sig = setup_sigma_graph(cg, WDT_HASEFFECT, pe,
                            custom_node_labels={WD["EFFECT_EVENT"]: "Effect Event", cause_ent: "Cause Event"},
                            custom_edge_labels={(WD["EFFECT_EVENT"], prediction_ent): "Predicted "},
                            custom_node_sizes={WD["EFFECT_EVENT"]: 5, cause_ent: 5, }, )
    return sig


def get_paths_on_case(c: SimilarCauseEffectChoice, kg: Graph,
                      target_prop: URIRef, path) -> Tuple[Graph, List[List[Tuple[URIRef, URIRef]]]]:
    case_graph = Graph()
    case_graph.bind("wd", WD)
    case_graph.bind("wdt", WDT)

    # try following the path for all of the cause event's properties
    all_path_edges = []
    for o in kg.objects(subject=c.effect, predicate=target_prop):
        path_edges = []
        path_trips = query_for_path(kg, path, c.cause, o, [])
        for t in path_trips:
            case_graph.add(t)
            path_edges.append((t[0], t[2]))
        all_path_edges.append(path_edges)

    return case_graph, all_path_edges


def show_paths_on_cases(cs: List[SimilarCauseEffectChoice], kg: Graph,
                        connecting_prop: URIRef, cause_props: List[URIRef],
                        effect_props: List[URIRef], workaround_country: URIRef,
                        target_prop: URIRef, path) -> Sigma:
    cg = new_collective_make_case_graph(cs, kg, connecting_prop, cause_props, effect_props, workaround_country)
    all_path_edges = []
    for c in cs:
        path_cg, ap_edges = get_paths_on_case(c, kg, target_prop, path)
        cg += path_cg
        all_path_edges += ap_edges

    sig = setup_sigma_graph(cg, WDT_HASEFFECT, all_path_edges)
    return sig

