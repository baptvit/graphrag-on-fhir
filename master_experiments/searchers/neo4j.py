import time
from abc import ABC, abstractmethod
from typing import List

from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from neo4j import GraphDatabase

import config
from master_experiments.healthcare.tracing import write_json_to_file


class SearchStrategy(ABC):
    def __init__(self):
        self.consumer_id: str = config.CONSUMER_ID

    @abstractmethod
    def execute(self, searcher: "Neo4jResourceSearcher", *args, **kwargs) -> List[str]:
        pass


class LexicalSearch0HopStrategy(SearchStrategy):
    def execute(
        self, searcher: "Neo4jResourceSearcher", resource_type: str
    ) -> List[str]:
        query = (
            """
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower('"""
            + resource_type
            + """') AND toLower(n.consumer_id) CONTAINS toLower('"""
            + self.consumer_id
            + """')
                RETURN "{'Main health record': '" + n.text + "'}\n" AS text
            """
        )
        return searcher._execute_query_with_timing(query)


class LexicalSearch1HopStrategy(SearchStrategy):
    def execute(
        self, searcher: "Neo4jResourceSearcher", resource_type: str
    ) -> List[str]:
        query = (
            """MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower('"""
            + resource_type
            + """') AND toLower(n.consumer_id) CONTAINS toLower('"""
            + self.consumer_id
            + """')
            WITH n
            CALL {
                WITH n
                MATCH (n)<-[r]->(neighbor)
                WHERE NOT type(r) IN ['subject', 'patient'] AND NOT neighbor:Date
                RETURN collect({neighbor_text: neighbor.text, relationship: type(r)}) AS neighbors
            }
           RETURN "{'Main health record': '" + n.text + "', 'Related health records': [" + 
                reduce(s = "", neighbor IN neighbors | s + "('Relationship' :'" + neighbor.relationship + "', 'Health Record' :'" + neighbor.neighbor_text + "'), ") + "]}" AS text
                """
        )
        print(query)
        return searcher._execute_query_with_timing(query)


class LexicalSearch2HopStrategy(SearchStrategy):
    def execute(
        self, searcher: "Neo4jResourceSearcher", resource_type: str
    ) -> List[str]:
        query = (
            """MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower('"""
            + resource_type
            + """') AND toLower(n.consumer_id) CONTAINS toLower('"""
            + self.consumer_id
            + """')
        WITH n
        CALL {
            WITH n
            MATCH (n)<-[r]->(neighbor)
            WHERE NOT type(r) IN ['subject', 'patient'] AND NOT neighbor:Date
            RETURN collect(DISTINCT {neighbor_text: neighbor.text, relationship: type(r)}) AS neighbors
        }
        CALL {
            WITH n
            MATCH (n)<-[r1]->(neighbor1)<-[r2]->(neighbor2)
            WHERE NOT type(r1) IN ['subject', 'patient'] AND NOT neighbor1:Date
            AND NOT type(r2) IN ['subject', 'patient'] AND NOT neighbor2:Date
            RETURN collect(DISTINCT {neighbor_text: neighbor2.text, relationship: type(r2)}) AS neighbors2
        }
        WITH n, neighbors, neighbors2, 
            apoc.coll.toSet(neighbors + neighbors2) AS all_neighbors
        RETURN "{'Main health record': '" + n.text + "', 'Related health records': [" + 
                        reduce(s = "", neighbor IN all_neighbors | s + "('Relationship' :'" + neighbor.relationship + "', 'Health Record' :'" + neighbor.neighbor_text + "'), ") + "]}" AS text
    """
        )
        print(query)
        return searcher._execute_query_with_timing(query)


class LexicalSearch3HopStrategy(SearchStrategy):
    def execute(
        self, searcher: "Neo4jResourceSearcher", resource_type: str
    ) -> List[str]:
        query = (
            """MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower('"""
            + resource_type
            + """') AND toLower(n.consumer_id) CONTAINS toLower('"""
            + self.consumer_id
            + """')
        WITH n
CALL {
    WITH n
    MATCH (n)<-[r]->(neighbor)
    WHERE NOT type(r) IN ['subject', 'patient'] AND NOT neighbor:Date
    RETURN collect(DISTINCT {neighbor_text: neighbor.text, relationship: type(r)}) AS neighbors
}
CALL {
    WITH n
    MATCH (n)<-[r1]->(neighbor1)<-[r2]->(neighbor2)
    WHERE NOT type(r1) IN ['subject', 'patient'] AND NOT neighbor1:Date
      AND NOT type(r2) IN ['subject', 'patient'] AND NOT neighbor2:Date
    RETURN collect(DISTINCT {neighbor_text: neighbor2.text, relationship: type(r2)}) AS neighbors2
}
CALL {
    WITH n
    MATCH (n)<-[r1]->(neighbor1)<-[r2]->(neighbor2)<-[r3]->(neighbor3)
    WHERE NOT type(r1) IN ['subject', 'patient'] AND NOT neighbor1:Date
      AND NOT type(r2) IN ['subject', 'patient'] AND NOT neighbor2:Date
      AND NOT type(r3) IN ['subject', 'patient'] AND NOT neighbor3:Date
    RETURN collect(DISTINCT {neighbor_text: neighbor3.text, relationship: type(r3)}) AS neighbors3
}
WITH n, neighbors, neighbors2, neighbors3,
     apoc.coll.toSet(neighbors + neighbors2 + neighbors3) AS all_neighbors
RETURN "{'Main health record': '" + n.text + "', 'Related health records': [" + 
    reduce(s = "", neighbor IN all_neighbors | s + "('Relationship' :'" + neighbor.relationship + "', 'Health Record' :'" + neighbor.neighbor_text + "'), ") + "]}" AS text
    """
        )
        print(query)
        return searcher._execute_query_with_timing(query)


class SimilaritySearch0HopStrategy(SearchStrategy):
    def execute(self, searcher: "Neo4jResourceSearcher", input_text: str) -> List[str]:
        query = (
            """MATCH (node)
                WHERE score >= """
            + str(searcher.similarity_threshold)
            + """
                WITH node
            RETURN "{'Main health record': '" + node.text + "'}\n" AS text
            , 1 AS score, {} AS metadata"""
        )

        vectorstore = Neo4jVector.from_existing_index(
            searcher.embedding_model,
            url=searcher.uri,
            username=searcher.user,
            password=searcher.password,
            database=searcher.database,
            index_name="fhir_text",
            retrieval_query=query,
        )

        start_time = time.time()
        results = vectorstore.similarity_search_with_score(
            query=input_text,
            k=searcher.k,
            score_threshold=searcher.similarity_threshold,
            filter={"consumer_id": self.consumer_id},
        )
        searcher._log_query_timing(start_time, "Similarity Search 0-Hop")
        return results


class SimilaritySearch1HopStrategy(SearchStrategy):
    def execute(self, searcher: "Neo4jResourceSearcher", input_text: str) -> List[str]:
        query = (
            """MATCH (node)
                WHERE score >= """
            + str(searcher.similarity_threshold)
            + """
                WITH node
                CALL {
                WITH node
                MATCH (node)<-[r]->(neighbor)
                WHERE NOT type(r) IN ['subject', 'patient']
                RETURN collect({neighbor_text: neighbor.text, relationship: type(r)}) AS neighbors
            }
            RETURN "{'Main health record': '" + node.text + "', 'Related health records': [" + 
                reduce(s = "", neighbor IN neighbors | s + "('Relationship' :'" + neighbor.relationship + "', 'Health Record' :'" + neighbor.neighbor_text + "'), ") + "]}\n" AS text
            , 1 AS score, {} AS metadata"""
        )
        vectorstore = Neo4jVector.from_existing_index(
            searcher.embedding_model,
            url=searcher.uri,
            username=searcher.user,
            password=searcher.password,
            database=searcher.database,
            index_name="fhir_text",
            retrieval_query=query,
        )

        start_time = time.time()
        results = vectorstore.similarity_search_with_score(
            query=input_text,
            k=searcher.k,
            score_threshold=searcher.similarity_threshold,
            filter={"consumer_id": self.consumer_id},
        )
        searcher._log_query_timing(start_time, "Similarity Search 1-Hop")
        return results


class SimilaritySearch2HopStrategy(SearchStrategy):
    def execute(self, searcher: "Neo4jResourceSearcher", input_text: str) -> List[str]:
        query = (
            """MATCH (node)
                WHERE score >= """
            + str(searcher.similarity_threshold)
            + """
        WITH node
        CALL {
            WITH node
            MATCH (node)<-[r]->(neighbor)
            WHERE NOT type(r) IN ['subject', 'patient'] AND NOT neighbor:Date
            RETURN collect(DISTINCT {neighbor_text: neighbor.text, relationship: type(r)}) AS neighbors
        }
        CALL {
            WITH node
            MATCH (node)<-[r1]->(neighbor1)<-[r2]->(neighbor2)
            WHERE NOT type(r1) IN ['subject', 'patient'] AND NOT neighbor1:Date
            AND NOT type(r2) IN ['subject', 'patient'] AND NOT neighbor2:Date
            RETURN collect(DISTINCT {neighbor_text: neighbor2.text, relationship: type(r2)}) AS neighbors2
        }
        WITH node, neighbors, neighbors2, 
            apoc.coll.toSet(neighbors + neighbors2) AS all_neighbors
        RETURN "{'Main health record': '" + node.text + "', 'Related health records': [" + 
                        reduce(s = "", neighbor IN all_neighbors | s + "('Relationship' :'" + neighbor.relationship + "', 'Health Record' :'" + neighbor.neighbor_text + "'), ") + "]}" AS text
            , 1 AS score, {} AS metadata"""
        )
        vectorstore = Neo4jVector.from_existing_index(
            searcher.embedding_model,
            url=searcher.uri,
            username=searcher.user,
            password=searcher.password,
            database=searcher.database,
            index_name="fhir_text",
            retrieval_query=query,
        )

        start_time = time.time()
        results = vectorstore.similarity_search_with_score(
            query=input_text,
            k=searcher.k,
            score_threshold=searcher.similarity_threshold,
            filter={"consumer_id": self.consumer_id},
        )
        searcher._log_query_timing(start_time, "Similarity Search 2-Hop")
        return results


class Neo4jResourceSearcher:
    def __init__(self, strategy: SearchStrategy):
        self.uri = config.NEO4J_URI
        self.user = config.NEO4J_USER
        self.password = config.NEO4J_PASSWORD
        self.database = config.NEO4J_DATABASE
        self.driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password), database=self.database
        )
        self.embedding_model = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
        self.strategy = strategy

        self.similarity_threshold = config.SIMILARITY_THRESHOLD
        self.k = config.K

    def close(self):
        self.driver.close()

    def search(self, *args, **kwargs):
        return self.strategy.execute(self, *args, **kwargs)

    def _execute_query_with_timing(self, query: str) -> List[str]:
        start_time = time.time()
        with self.driver.session() as session:
            result = session.run(query)
            records = [record["text"] for record in result]
        self._log_query_timing(start_time, query)
        return records

    def _log_query_timing(self, start_time: float, query: str):
        latency = time.time() - start_time
        write_json_to_file(
            {
                "latency_s": [latency],
                "query": [str(query)],
                "similarity_threshold": [self.similarity_threshold],
                "k": [self.k],
                "embedding_model": [self.embedding_model.model_name],
            },
            "search_step",
        )
