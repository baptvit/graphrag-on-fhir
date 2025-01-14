import json
import re

from helpers.FHIR_to_graph_v1 import resource_to_edges, resource_to_node


def process_bundle(bundle_file_name):
    nodes = []
    edges = []
    dates = set()  # set is used here to make sure dates are unique
    with open(bundle_file_name) as raw:
        bundle = json.load(raw)
        for entry in bundle["entry"]:
            resource_type = entry["resource"]["resourceType"]
            if resource_type != "Provenance":
                create_graph_from_resources(entry, nodes, edges, dates)
    return nodes, edges, dates


def create_graph_from_resources(entry, nodes, edges, dates):
    # Generate the Cypher for creating the resource node
    nodes.append(
        resource_to_node(
            entry["resource"], bundle_file_name.split("/")[-1].replace(".json", "")
        )
    )
    # Generate the Cypher for creating the reference & date edges and capture dates
    node_edges, node_dates = resource_to_edges(entry["resource"])
    edges += node_edges
    dates.update(node_dates)


def create_nodes_to_graph(nodes):
    for node in nodes:
        try:
            graph.query(node)
        except:
            print(f"Failed to create edge: {edge}")


def create_dates_to_graph(dates):
    date_pattern = re.compile(r"([0-9]+)/([0-9]+)/([0-9]+)")
    for date in dates:
        date_parts = date_pattern.findall(date)[0]
        cypher_date = f"{date_parts[2]}-{date_parts[0]}-{date_parts[1]}"
        cypher = f'CREATE (:Date {{name:"{date}", id: "{date}", date: date("{cypher_date}"), text:"{date}"}})'
        graph.query(cypher)


def create_edges_to_graph(edges):
    for edge in edges:
        try:
            graph.query(edge)
        except:
            print(f"Failed to create edge: {edge}")


if __name__ == "__main__":
    synthea_bundles = glob.glob(
        "/home/baptvit/repositories/fhir_based_gen_ai_research/fhir_rag/fhir_data/stanford_llm_on_fhir/Beatris270_Bogan287_5b3645de-a2d0-d016-0839-bab3757c4c58.json"
    )
    synthea_bundles = synthea_bundles[0:1]
    synthea_bundles.sort()

    nodes_total, edges_total, dates_total = process_bundle(synthea_bundles[0])

    # Create the nodes for resources
    create_nodes_to_graph(nodes_total)
    # Create the nodes for dates
    create_dates_to_graph(dates_total)
    # Create the edges
    create_edges_to_graph(edges_total)
