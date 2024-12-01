import os
import uuid
import json
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Any
from copy import deepcopy
import pytz
import hydra

from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage
from aiflows.prompt_template import JinjaPrompt
from aiflows.backends.llm_lite import LiteLLMBackend
from langchain_cohere import CohereEmbeddings
import chromadb
from chromadb.config import Settings
from neo4j import GraphDatabase
from rank_bm25 import BM25Okapi
import os
import uuid
import json
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Any
from copy import deepcopy
import numpy as np

import hydra
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage
from aiflows.prompt_template import JinjaPrompt
from aiflows.backends.llm_lite import LiteLLMBackend
from langchain_cohere import CohereEmbeddings
import chromadb
from chromadb.config import Settings
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

EXTRACT_ENTITIES_PROMPT = """
You are an advanced algorithm designed to extract structured information from text to construct knowledge graphs. Your goal is to capture comprehensive information while maintaining accuracy.

Instructions:
1. Extract only explicitly stated information from the text.
2. Identify nodes (entities/concepts), their types, and relationships.
3. Use the provided USER_ID as the source node for any self-references (I, me, my, etc.) in user messages.
4. Provide the output strictly in the following JSON format without any additional text or explanation:

{
    "entities": [
        {
            "event": "ADD" or "UPDATE" or "NONE",
            "source": "source node identifier",
            "source_type": "type of the source node",
            "relationship": "relationship between source and destination",
            "destination": "destination node identifier",
            "destination_type": "type of the destination node"
        },
        ...
    ]
}

Ensure that the JSON is properly formatted and parsable.

Conversation:
{{conversation}}
"""



class OpenAIEmbeddingFunction:
    def __init__(self, openai_embedding_model):
        self.embedding_model = openai_embedding_model

    def __call__(self, input):
        return self.embedding_model.embed_documents(input)

class PersonalMemoryFlow(AtomicFlow):
    """
    PersonalMemoryFlow for managing memory updates and retrievals with vector store and graph capabilities.

    - Extracts facts and entities from conversations.
    - Updates both vector store and graph database with the extracted information.
    - Fetches relevant memories from both vector store and graph database.
    """

    def __init__(
        self,
        llm_backend: LiteLLMBackend,
        vector_backend: LiteLLMBackend,
        graph_config: Dict[str, Any],
        update_prompt_template: JinjaPrompt,
        fact_retrieval_prompt_template: JinjaPrompt,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.llm_backend = llm_backend
        self.vector_backend = vector_backend
        self.update_prompt_template = update_prompt_template
        self.fact_retrieval_prompt_template = fact_retrieval_prompt_template
        self.embeddings_model = CohereEmbeddings(model="embed-english-v3.0")
        self.initialize_collection()
        self.embeddings = self.get_embeddings_model()

        self.graph_driver = GraphDatabase.driver(
            graph_config['uri'],
            auth=(
                graph_config['user'],
                graph_config['password']
            )
        )

        self.set_up_flow_state()

    def set_up_flow_state(self):
        super().set_up_flow_state()
        self.flow_state["running_context"] = []

    @classmethod
    def instantiate_from_config(cls, config: Dict) -> 'PersonalMemoryFlow':
        flow_config = deepcopy(config)

        kwargs = {"flow_config": flow_config}
        kwargs["llm_backend"] = hydra.utils.instantiate(flow_config["llm_backend"], _convert_="partial")
        kwargs["vector_backend"] = hydra.utils.instantiate(flow_config["vector_backend"], _convert_="partial")
        kwargs["update_prompt_template"] = hydra.utils.instantiate(
            flow_config["update_prompt_template"], _convert_="partial"
        )
        kwargs["fact_retrieval_prompt_template"] = hydra.utils.instantiate(
            flow_config["fact_retrieval_prompt_template"], _convert_="partial"
        )
        kwargs["graph_config"] = flow_config.get("graph_config", {})

        return cls(**kwargs)

    def initialize_collection(self):
        """
        Initialize the ChromaDB collection.
        If no data exists, prepare an empty collection for future usage.
        """
        settings = Settings(
            anonymized_telemetry=False,
            persist_directory=self.flow_config.get("persist_directory", "./chroma_db_dir"),
            is_persistent=True
        )
        self.client = chromadb.Client(settings)

        self.collection_name = self.flow_config.get("collection_name", "personal_memory_collection")

        embedding_function = OpenAIEmbeddingFunction(self.get_embeddings_model())

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_function
        )

        collection_count = self.collection.count()
        if collection_count == 0:
            logger.info(f"Collection '{self.collection_name}' is empty. Ready for new memories.")
        else:
            logger.info(f"Collection '{self.collection_name}' contains {collection_count} memories.")

        self.flow_state["db_initialized"] = True

    def run(self, input_message: FlowMessage):
        input_data = input_message.data
        operation = input_data.get("operation")
        content = input_data.get("content")

        user_id = input_data.get("user_id")
        if not user_id:
            response = {"error": "User ID is required for this operation."}
            reply_message = self.package_output_message(
                input_message=input_message,
                response=response,
            )
            self.send_message(reply_message)
            return

        if operation == "update":
            facts = self.extract_facts(content["content"])
            entities = self.extract_entities(content["content"], user_id)
            vector_response = self.update_memory(facts, user_id)
            graph_response = self.update_graph_memory(entities, user_id)
            response = {
                "vector_response": vector_response,
                "graph_response": graph_response
            }
        elif operation == "fetch":
            vector_results = self.fetch_memory(content["query"], user_id)
            graph_results = self.fetch_graph_memory(content["query"], user_id)
            response = {
                "vector_results": vector_results,
                "graph_results": graph_results
            }
        else:
            response = {"error": f"Unsupported operation '{operation}'."}

        reply_message = self.package_output_message(
            input_message=input_message,
            response=response,
        )
        self.send_message(reply_message)

    def extract_facts(self, conversation: str) -> List[str]:
        """Extract facts from a conversation using the fact retrieval prompt."""
        update_input = {
            "conversation": conversation
        }
        prompt_text = self.fact_retrieval_prompt_template.format(**update_input)
        messages = [{"role": "user", "content": prompt_text}]

        response = self.llm_backend(messages=messages)

        try:
            content = response[0]["content"]
            facts_json = json.loads(content)
            return facts_json.get("facts", [])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse facts from LLM response: {e}")
            return []
        except Exception as e:
            logger.error(f"An error occurred while extracting facts: {e}")
            return []

    def extract_entities(self, conversation: str, user_id: str) -> List[Dict[str, Any]]:
        """Extract entities and relationships from the conversation for graph storage."""
        prompt_text = EXTRACT_ENTITIES_PROMPT.replace("{{conversation}}", conversation).replace("USER_ID", user_id)
        messages = [{"role": "user", "content": prompt_text}]

        response = self.llm_backend(messages=messages)

        try:
            content = response[0]["content"]
            entities_json = json.loads(content)
            return entities_json.get("entities", [])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse entities from LLM response: {e}")
            return []
        except Exception as e:
            logger.error(f"An error occurred while extracting entities: {e}")
            return []

    

    def update_graph_memory(self, entities: List[Dict[str, Any]], user_id: str) -> Dict[str, Any]:
        """Update the graph database with extracted entities."""
        with self.graph_driver.session() as session:
            for entity in entities:
                event = entity.get("event")
                if event == "ADD":
                    source = entity["source"].lower().replace(" ", "_")
                    source_type = entity["source_type"].lower().replace(" ", "_")
                    relationship = entity["relationship"].upper().replace(" ", "_")
                    target = entity["destination"].lower().replace(" ", "_")
                    target_type = entity["destination_type"].lower().replace(" ", "_")

                    source_embedding = self.embeddings_model.embed_documents([source])[0]
                    target_embedding = self.embeddings_model.embed_documents([target])[0]

                    query = """
                    MERGE (s:{source_type} {{name: $source, user_id: $user_id}})
                    ON CREATE SET s.embedding = $source_embedding
                    MERGE (t:{target_type} {{name: $target, user_id: $user_id}})
                    ON CREATE SET t.embedding = $target_embedding
                    MERGE (s)-[r:{relationship}]->(t)
                    ON CREATE SET r.created = timestamp()
                    """.format(
                        source_type=source_type.capitalize(),
                        target_type=target_type.capitalize(),
                        relationship=relationship
                    )
                    session.run(query, source=source, target=target, user_id=user_id,
                                source_embedding=source_embedding, target_embedding=target_embedding)
                elif event == "UPDATE":
                    source_type = entity["source_type"].lower()
                    target_type = entity["destination_type"].lower()

                    self._update_relationship(
                        source=entity["source"],
                        target=entity["destination"],
                        relationship=entity["relationship"],
                        user_id=user_id,
                    )
                elif event == "NONE":
                    continue
                else:
                    logger.warning(f"Unrecognized event type: {event}")
        return {"result": "Graph memory updated successfully."}

    def _update_relationship(self, source, target, relationship, user_id):
        """
        Update or create a relationship between two nodes in the graph.

        Args:
            source (str): The name of the source node.
            target (str): The name of the target node.
            relationship (str): The type of the relationship.
            user_id (str): The user ID.
        """
        logger.info(f"Updating relationship: {source} -{relationship}-> {target}")
        relationship = relationship.upper().replace(" ", "_")

        with self.graph_driver.session() as session:
            check_and_create_query = """
            MERGE (n1 {name: $source, user_id: $user_id})
            MERGE (n2 {name: $target, user_id: $user_id})
            """
            session.run(
                check_and_create_query,
                source=source, target=target, user_id=user_id,
            )

            delete_query = """
            MATCH (n1 {name: $source, user_id: $user_id})-[r]->(n2 {name: $target, user_id: $user_id})
            DELETE r
            """
            session.run(
                delete_query,
                source=source, target=target, user_id=user_id,
            )

            create_query = f"""
            MATCH (n1 {{name: $source, user_id: $user_id}}), (n2 {{name: $target, user_id: $user_id}})
            CREATE (n1)-[r:{relationship}]->(n2)
            RETURN n1, r, n2
            """
            result = session.run(
                create_query,
                source=source, target=target, user_id=user_id,
            )

            if not result:
                raise Exception(f"Failed to update or create relationship between {source} and {target}")

    def retrieve_existing_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve existing memories from the ChromaDB collection."""
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=["documents", "metadatas", "ids"]
            )

            if not results or not results.get("documents"):
                logger.info("No existing memories found in the collection.")
                return []

            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            ids = results.get("ids", [])

            combined_results = [
                {
                    "id": ids[i],
                    "text": documents[i],
                    "metadata": metadatas[i] if i < len(metadatas) else {}
                }
                for i in range(len(documents))
            ]
            return combined_results
        except Exception as e:
            logger.error(f"Error retrieving existing memories: {e}")
            return []

    

   

    def fetch_graph_memory(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """Fetch relevant entities from the graph database based on the query."""
        query_embedding = self.embeddings_model.embed_documents([query])[0]

        with self.graph_driver.session() as session:
            result = session.run("""
            MATCH (n)
            WHERE n.embedding IS NOT NULL AND n.user_id = $user_id
            RETURN n.name AS name, n.embedding AS embedding
            """, user_id=user_id)
            nodes = result.data()
            if not nodes:
                return []

            node_names = [node['name'] for node in nodes]
            node_embeddings = [node['embedding'] for node in nodes]

            similarities = self.compute_cosine_similarities(query_embedding, node_embeddings)

            threshold = 0.4

            similar_nodes = [node_names[i] for i, sim in enumerate(similarities) if sim >= threshold]

            if not similar_nodes:
                return []

            params = {'user_id': user_id}
            for idx, name in enumerate(similar_nodes):
                params[f'name{idx}'] = name
            names_placeholders = ', '.join([f'$name{idx}' for idx in range(len(similar_nodes))])

            cypher_query = f"""
            MATCH (s)-[r]->(t)
            WHERE s.user_id = $user_id AND t.user_id = $user_id
            AND (s.name IN [{names_placeholders}] OR t.name IN [{names_placeholders}])
            RETURN s.name AS source, type(r) AS relationship, t.name AS target
            """

            result = session.run(cypher_query, **params)
            records = result.data()
            return records

    def compute_cosine_similarities(self, query_embedding, node_embeddings):
        query_vec = np.array(query_embedding)
        node_vecs = np.array(node_embeddings)
        dot_products = np.dot(node_vecs, query_vec)
        query_norm = np.linalg.norm(query_vec)
        node_norms = np.linalg.norm(node_vecs, axis=1)
        similarities = dot_products / (node_norms * query_norm + 1e-8)
        return similarities.tolist()

    def get_embeddings_model(self):
        api_information = self.vector_backend.get_key()
        if api_information.backend_used == "cohere":
            embeddings = CohereEmbeddings(model="embed-english-v3.0")
        else:
            embeddings = CohereEmbeddings(model="embed-english-v3.0")
        return embeddings

    def __del__(self):
        if hasattr(self, 'graph_driver'):
            self.graph_driver.close()
