import concurrent.futures
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from master_experiments.ingestion.helpers.FHIR_to_graph import (
    resource_to_edges,
    resource_to_node,
)
from master_experiments.ingestion.helpers.neo4j_graph import Graph


def process_bundle(bundle_file_name, graph):
    edges = []
    dates = set()  # set is used here to make sure dates are unique

    with open(bundle_file_name) as raw:
        bundle = json.load(raw)
        with ThreadPoolExecutor(50) as executor:
            futures = [
                executor.submit(
                    create_graph_from_resources, entry, bundle_file_name, graph
                )
                for entry in bundle["entry"]
                if entry["resource"]["resourceType"] != "Provenance"
            ]

        for future in concurrent.futures.as_completed(futures):
            try:
                edge, date = future.result()
                edges += edge
                dates.update(date)
            except Exception as e:
                print(f"Error processing entry: {e}")

    return edges, dates


def create_graph_from_resources(entry, bundle_file_name, graph):
    edges, dates = [], set()
    # Generate the Cypher for creating the resource node
    node = resource_to_node(
        entry["resource"], bundle_file_name.split("/")[-1].replace(".json", "")
    )

    # Generate the Cypher for creating the reference & date edges and capture dates
    node_edges, node_dates = resource_to_edges(entry["resource"])
    edges += node_edges
    dates.update(node_dates)

    create_nodes_to_graph([node], graph)
    # create_dates_to_graph(list(dates), graph)
    # create_edges_to_graph(edges, graph)
    time.sleep(0.5)
    return edges, dates


def create_nodes_to_graph(nodes, graph):
    for node in nodes:
        try:
            graph.query(node)
            print("Node created")

        except Exception as e:
            if e == "local variable 'e' referenced before assignment":
                time.sleep(1)
                graph.query(node)
                print(f"Node created after retry: {e}")
            else:
                print(node)
                print("\n\n\n\n\n\n")
                print(f"Failed to create node: {e}")


def create_dates_to_graph(dates, graph):
    date_pattern = re.compile(r"([0-9]+)/([0-9]+)/([0-9]+)")
    for date in dates:
        try:
            date_parts = date_pattern.findall(date)[0]
            cypher_date = f"{date_parts[2]}-{date_parts[0]}-{date_parts[1]}"
            cypher = f'CREATE (:Date {{name:"{date}", id: "{date}", date: date("{cypher_date}"), text:"{date}"}})'
            graph.query(cypher)
            print("Date created")
        except Exception as e:
            print(f"Failed to create date: {e}")


def create_edges_to_graph(edges, graph):
    for edge in edges:
        try:
            graph.query(edge)
            print("Edge created")
        except Exception as e:
            print(f"Failed to create edge: {e}")


if __name__ == "__main__":
    NEO4J_URI = os.getenv("FHIR_GRAPH_URL", "neo4j://192.168.2.129:7687")
    USERNAME = os.getenv("FHIR_GRAPH_USER", "neo4j")
    PASSWORD = os.getenv("FHIR_GRAPH_PASSWORD", "password")
    DATABASE = os.getenv("FHIR_GRAPH_DATABASE", "neo4j")

    graph = Graph(NEO4J_URI, USERNAME, PASSWORD, DATABASE)

    graph.query(
        """
        CREATE VECTOR INDEX fhir_text IF NOT EXISTS
        FOR (n:resource)
        ON n.embedding
        OPTIONS { indexConfig: {
        `vector.dimensions`: 384,
        `vector.similarity_function`: 'cosine'
        }
        }
        """
    )
    synthea_bundles = glob.glob("master_experiments/fhir_data/data/*.json")

    synthea_bundles = synthea_bundles[0:1]
    # synthea_bundles.sort()

    for bundle in synthea_bundles:
        start = time.time()
        print("Start ingesting bundle: ", str(bundle))
        edges, dates = process_bundle(bundle, graph)

        # # Create the nodes for resources
        # print("Adding nodes: ", str(bundle))
        # create_nodes_to_graph(nodes_total, graph)
        # Create the nodes for dates
        # print("Adding Dates: ", str(bundle))
        # create_dates_to_graph(dates, graph)
        # Create the edges
        print("Adding Edges: ", str(bundle))
        create_edges_to_graph(edges, graph)

        end = time.time()
        print("Took in time", end - start)
