
import os
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime
import hashlib
import uuid
import base64

from copy import deepcopy
from langchain_cohere import CohereEmbeddings

from aiflows.messages import FlowMessage
from aiflows.base_flows import AtomicFlow
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import hydra

import chromadb
from chromadb.config import Settings
import numpy as np  

import logging
from pydantic import BaseModel


class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[Dict]
    document: Optional[str]


class OpenAIEmbeddingFunction:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Embeds a list of input texts and returns their embeddings.

        Args:
            input (List[str]): List of texts/documents to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        return self.embedding_model.embed_documents(input)


def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj


class ChromaDBFlow(AtomicFlow):
    """
    An atomic flow that interfaces with a Chroma vector store for comprehensive vector operations.

    Supports operations:
    - Data Management: insert, update, delete, get
    - Collection Management: list_collections, delete_collection, collection_info
    - Vector Management: list_vectors
    - Retrieval Operations: similarity_search, similarity_search_with_score,
      similarity_search_with_relevance_scores, max_marginal_relevance_search
    - Image Operations: add_images, similarity_search_by_image, similarity_search_by_image_with_relevance_score
    """

    def __init__(self, backend: Any, **kwargs):
        super().__init__(**kwargs)
        self.backend = backend
        self.logger = logging.getLogger(__name__)
        self.flow_state["db_initialized"] = False
        self.flow_state['sex'] = False
        self.collection = None  

    def set_up_flow_state(self):
        super().set_up_flow_state()
        self.flow_state["db_initialized"] = False
        self.collection = None  

    @classmethod
    def _set_up_backend(cls, config: Dict) -> Dict:
        """Instantiate the backend from a configuration file."""
        kwargs = {}
        kwargs["backend"] = hydra.utils.instantiate(config['backend'], _convert_="partial")
        return kwargs

    @classmethod
    def instantiate_from_config(cls, config: Dict) -> 'ChromaDBFlow':
        """Instantiate the flow from a configuration file."""
        flow_config = deepcopy(config)
        kwargs = {"flow_config": flow_config}
        kwargs.update(cls._set_up_backend(flow_config))
        return cls(**kwargs)

    def get_embeddings_model(self):
        """Retrieve the embeddings model using the backend."""
        api_information = self.backend.get_key()
        if api_information.backend_used == "cohere":
            embeddings = CohereEmbeddings(model="embed-english-v3.0")
        else:
            embeddings = CohereEmbeddings(model="embed-english-v3.0") 
        return embeddings

    def initialize_db(self):
        """Initialize the ChromaDB client and collection."""
        if True:
            settings = Settings(
                anonymized_telemetry=False,
                persist_directory=self.flow_config.get("persist_directory", "./chroma_db_dir"),
                is_persistent=True
            )
            self.client = chromadb.Client(settings)

            self.collection_name = self.flow_config.get("collection_name", "default_collection")

            embedding_model = self.get_embeddings_model()
            embedding_function = OpenAIEmbeddingFunction(embedding_model)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_function  
            )

            collection_count = self.collection.count()
            if collection_count == 0:
                paths_to_data = self.flow_config.get("paths_to_data", [])
                if paths_to_data:
                    full_docs = []
                    metadatas = []
                    text_splitter = CharacterTextSplitter(
                        chunk_size=self.flow_config.get("chunk_size", 1000),
                        chunk_overlap=self.flow_config.get("chunk_overlap", 0),
                        separator=self.flow_config.get("separator", "\n\n")
                    )
                    for path in paths_to_data:
                        loader = TextLoader(path)
                        documents = loader.load()
                        docs = text_splitter.split_documents(documents)
                        full_docs.extend(docs)
                        for doc in docs:
                            metadata = {
                                'source': path,
                                'hash': hashlib.md5(doc.page_content.encode('utf-8')).hexdigest(),
                                'created_at': datetime.utcnow().isoformat(),
                                'updated_at': datetime.utcnow().isoformat()
                            }
                            metadatas.append(metadata)
                    texts = [doc.page_content for doc in full_docs]
                    embeddings = embedding_model.embed_documents(texts)
                    ids = [str(uuid.uuid4()) for _ in texts]
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metadatas
                    )
                    self.logger.info(f"Loaded {len(texts)} documents into the collection.")
            else:
                self.logger.info(f"Collection '{self.collection_name}' already contains data.")

            self.flow_state["db_initialized"] = True

    def _parse_output(self, data: Dict) -> List[OutputData]:
        """Parse ChromaDB query results into OutputData instances."""
        keys = ["ids", "distances", "metadatas", "documents"]
        values = {}

        for key in keys:
            value = data.get(key, [])
            if isinstance(value, list) and value and isinstance(value[0], list):
                value = value[0]
            values[key] = value

        ids = values.get('ids', [])
        distances = values.get('distances', [])
        metadatas = values.get('metadatas', [])
        documents = values.get('documents', [])

        max_length = max(len(v) for v in [ids, distances, metadatas, documents] if isinstance(v, list))

        result = []
        for i in range(max_length):
            entry = OutputData(
                id=ids[i] if i < len(ids) else None,
                score=distances[i] if i < len(distances) else None,
                payload=metadatas[i] if i < len(metadatas) else None,
                document=documents[i] if i < len(documents) else None,
            )
            result.append(entry)

        return result

    def run(self, input_message: FlowMessage):
        """Run the flow, supporting all vector store operations with ChromaDB."""
        self.initialize_db()
        input_data = input_message.data
        embeddings = self.get_embeddings_model()
        response = {}

        operation = input_data.get("operation")
        content = input_data.get("content", {})

        if operation not in [
            "insert", "search", "update", "delete", "get",
            "list_collections", "delete_collection", "collection_info", "list_vectors",
            "similarity_search", "similarity_search_with_score",
            "similarity_search_with_relevance_scores", "max_marginal_relevance_search",
            "add_images", "similarity_search_by_image", "similarity_search_by_image_with_relevance_score"
        ]:
            raise ValueError(f"Operation '{operation}' not supported")

        if operation == "insert":
            self.flow_state['sex'] = True
            self.handle_insert(content, embeddings, response)
        elif operation == "search":
            self.handle_search(content, embeddings, response)
        elif operation == "update":
            self.handle_update(content, embeddings, response)
        elif operation == "delete":
            self.handle_delete(content, response)
        elif operation == "get":
            self.handle_get(content, response)
        elif operation == "list_collections":
            print(self.flow_state['sex'])
            self.handle_list_collections(response)
        elif operation == "delete_collection":
            self.handle_delete_collection(content, response)
        elif operation == "collection_info":
            self.handle_collection_info(content, response)
        elif operation == "list_vectors":
            self.handle_list_vectors(content, response)
        elif operation == "similarity_search":
            self.handle_similarity_search(content, embeddings, response)
        elif operation == "similarity_search_with_score":
            self.handle_similarity_search_with_score(content, embeddings, response)
        elif operation == "similarity_search_with_relevance_scores":
            self.handle_similarity_search_with_relevance_scores(content, embeddings, response)
        elif operation == "max_marginal_relevance_search":
            self.handle_max_marginal_relevance_search(content, embeddings, response)
        elif operation == "add_images":
            self.handle_add_images(content, embeddings, response)
        elif operation == "similarity_search_by_image":
            self.handle_similarity_search_by_image(content, embeddings, response)
        elif operation == "similarity_search_by_image_with_relevance_score":
            self.handle_similarity_search_by_image_with_relevance_score(content, embeddings, response)
        else:
            raise ValueError(f"Unknown operation '{operation}'")

        reply = self.package_output_message(
            input_message=input_message,
            response=response
        )
        self.send_message(reply)



    def handle_insert(self, content: Dict, embeddings, response: Dict):
        """Handle the insert operation."""
        vectors = content.get("vectors")
        payloads = content.get("payloads")
        ids = content.get("ids")
        if vectors is None:
            raise ValueError("Content must include 'vectors' for insert operation")

        if not isinstance(vectors, list):
            vectors = [vectors]

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        elif not isinstance(ids, list):
            ids = [ids]

        if payloads is None:
            payloads = [{} for _ in vectors]
        elif not isinstance(payloads, list):
            payloads = [payloads]

        for i, payload in enumerate(payloads):
            payload['hash'] = hashlib.md5(vectors[i].encode('utf-8')).hexdigest()
            payload['created_at'] = datetime.utcnow().isoformat()
            payload['updated_at'] = datetime.utcnow().isoformat()

        embeddings_list = embeddings.embed_documents(vectors)

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=vectors,  
            metadatas=payloads
        )
        response["result"] = f"Inserted {len(ids)} vectors."

    def handle_search(self, content: Dict, embeddings, response: Dict):
        query = content.get("query")
        limit = content.get("limit", 5)
        filters = content.get("filters")
        if query is None:
            raise ValueError("Content must include 'query' for search operation")

        if not isinstance(query, list):
            query = [query]

        query_embeddings = embeddings.embed_documents(query)

        results = self.collection.query(
            query_embeddings=query_embeddings,
            where=filters,
            n_results=limit,
            include=["embeddings", "documents", "metadatas", "distances"]  # Include fields
        )

        results = convert_numpy_to_list(results)

        parsed_results = self._parse_output(results)

        response["result"] = [result.dict() for result in parsed_results]

    def handle_update(self, content: Dict, embeddings, response: Dict):
        """Handle the update operation."""
        vector_id = content.get("id")
        vector = content.get("vector")
        payload = content.get("payload")
        if vector_id is None:
            raise ValueError("Content must include 'id' for update operation")

        embeddings_list = None
        if vector is not None:
            embeddings_list = embeddings.embed_documents([vector])

        if payload is None:
            payload = {}
        payload['updated_at'] = datetime.utcnow().isoformat()

        self.collection.update(
            ids=[vector_id],
            embeddings=embeddings_list,
            metadatas=[payload]
        )
        response["result"] = f"Updated vector with id {vector_id}."

    def handle_delete(self, content: Dict, response: Dict):
        """Handle the delete operation."""
        vector_id = content.get("id")
        if vector_id is None:
            raise ValueError("Content must include 'id' for delete operation")

        existing_ids = self.collection.get(ids=[vector_id])['ids']
        if not existing_ids:
            response["result"] = f"Vector with id {vector_id} does not exist."
        else:
            self.collection.delete(ids=[vector_id])
            response["result"] = f"Deleted vector with id {vector_id}."

    def handle_get(self, content: Dict, response: Dict):
        """Handle the get operation."""
        vector_id = content.get("id")
        if vector_id is None:
            raise ValueError("Content must include 'id' for get operation")
        result = self.collection.get(
            ids=[vector_id],
            include=["embeddings", "documents", "metadatas"]
        )

        result = convert_numpy_to_list(result)

        parsed_result = self._parse_output(result)

        response["result"] = [entry.dict() for entry in parsed_result]


    def handle_list_collections(self, response: Dict):
        """Handle listing all collections."""
        collections = self.client.list_collections()
        collection_names = [col.name for col in collections]
        response["result"] = collection_names

    def handle_delete_collection(self, content: Dict, response: Dict):
        """Handle deleting a specific collection."""
        collection_name = content.get("collection_name", self.collection_name)
        self.client.delete_collection(name=collection_name)
        response["result"] = f"Deleted collection '{collection_name}'."
        self.logger.info(f"Deleted collection '{collection_name}'.")
        self.flow_state["db_initialized"] = False
        self.collection = None
        self.logger.info(f"Reset flow state after deleting collection '{collection_name}'.")


    def handle_collection_info(self, content: Dict, response: Dict):
        """Handle retrieving information about a collection."""
        collection_name = content.get("collection_name", self.collection_name)
        try:
            collection = self.client.get_collection(name=collection_name)
            info = {
                "name": collection.name,
                "metadata": collection.metadata,
                "count": collection.count()
            }
            response["result"] = info
        except Exception as e:
            response["result"] = f"Error retrieving collection info: {str(e)}"

    def handle_list_vectors(self, content: Dict, response: Dict):
        """Handle listing vectors/documents in a collection."""
        filters = content.get("filters")
        limit = content.get("limit", 100)
        results = self.collection.get(
            where=filters,
            limit=limit,
            include=["embeddings", "documents", "metadatas"]
        )

        results = convert_numpy_to_list(results)

        parsed_results = self._parse_output(results)

        response["result"] = [result.dict() for result in parsed_results]


    def handle_similarity_search(self, content: Dict, embeddings, response: Dict):
        """Handle the similarity_search operation."""
        query = content.get("query")
        k = content.get("k", 4)
        filter = content.get("filter")
        if query is None:
            raise ValueError("Content must include 'query' for similarity_search operation")

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter,
            include=["documents", "metadatas", "distances"]
        )

        parsed_results = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            parsed_results.append({
                'document': doc,
                'metadata': metadata,
                'distance': distance
            })
        response["result"] = parsed_results

    def handle_similarity_search_with_score(self, content: Dict, embeddings, response: Dict):
        """Handle the similarity_search_with_score operation."""
        query = content.get("query")
        k = content.get("k", 4)
        filter = content.get("filter")
        if query is None:
            raise ValueError("Content must include 'query' for similarity_search_with_score operation")

        if self.backend.get_key().backend_used == "cohere":
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=filter,
                include=["documents", "metadatas", "distances"]
            )
        else:
            query_embedding = embeddings.embed_query(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                include=["documents", "metadatas", "distances"]
            )

        parsed_results = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            parsed_results.append({
                'document': doc,
                'metadata': metadata,
                'score': distance  
            })
        response["result"] = parsed_results

    
    def handle_similarity_search_with_relevance_scores(self, content: Dict, embeddings: CohereEmbeddings, response: Dict):
        """Handle the similarity_search_with_relevance_scores operation."""
        query = content.get("query")
        k = content.get("k", 4)
        score_threshold = content.get("score_threshold", 0.5)
        filter = content.get("filter")
        if query is None:
            raise ValueError("Content must include 'query' for similarity_search_with_relevance_scores operation")

        try:
            query_embedding = embeddings.embed_query(query)
            self.logger.debug(f"Query embedding: {query_embedding}")

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                include=["documents", "metadatas", "distances"]
            )
            self.logger.debug(f"Query results: {results}")

            relevance_scores = []
            for distance in results['distances'][0]:
                relevance_score = 1.0 - (distance / 2.0) 
                relevance_scores.append(relevance_score)
                self.logger.debug(f"Distance: {distance}, Relevance Score: {relevance_score}")

            parsed_results = []
            for doc, metadata, score in zip(
                results['documents'][0],
                results['metadatas'][0],
                relevance_scores
            ):
                self.logger.debug(f"Document: {doc}, Metadata: {metadata}, Relevance Score: {score}")
                if score >= score_threshold:
                    parsed_results.append({
                        'document': doc,
                        'metadata': metadata,
                        'relevance_score': score
                    })

            response["result"] = parsed_results
            self.logger.info(f"Performed similarity search with relevance scores for query: '{query}', k: {k}, threshold: {score_threshold}.")

        except Exception as e:
            self.logger.error(f"Error in similarity_search_with_relevance_scores: {str(e)}")
            response["error"] = f"Error in similarity_search_with_relevance_scores: {str(e)}"



    def handle_max_marginal_relevance_search(self, content: Dict, embeddings, response: Dict):
        """Handle the max_marginal_relevance_search operation."""
        query = content.get("query")
        k = content.get("k", 4)
        fetch_k = content.get("fetch_k", 20)
        lambda_mult = content.get("lambda_mult", 0.5)
        filter = content.get("filter")
        if query is None:
            raise ValueError("Content must include 'query' for max_marginal_relevance_search operation")

        embedding = embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=fetch_k,
            where=filter,
            include=["documents", "metadatas", "embeddings"]
        )

        embeddings_array = np.array(results['embeddings'][0], dtype=np.float32)
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]

        selected_indices = self.maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embeddings_array,
            k=k,
            lambda_mult=lambda_mult
        )

        selected_results = []
        for idx in selected_indices:
            selected_results.append({
                'document': documents[idx],
                'metadata': metadatas[idx]
            })
        response["result"] = selected_results


    def handle_add_images(self, content: Dict, embeddings, response: Dict):
        """Handle the add_images operation."""
        uris = content.get("uris")
        metadatas = content.get("metadatas")
        ids = content.get("ids")
        if uris is None:
            raise ValueError("Content must include 'uris' for add_images operation")

        if not isinstance(uris, list):
            uris = [uris]

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(uris))]
        elif not isinstance(ids, list):
            ids = [ids]

        if metadatas is None:
            metadatas = [{} for _ in uris]
        elif not isinstance(metadatas, list):
            metadatas = [metadatas]

        for i, metadata in enumerate(metadatas):
            metadata['hash'] = hashlib.md5(uris[i].encode('utf-8')).hexdigest()
            metadata['created_at'] = datetime.utcnow().isoformat()
            metadata['updated_at'] = datetime.utcnow().isoformat()

        encoded_images = [self.encode_image(uri) for uri in uris]

        if hasattr(embeddings, "embed_documents"):  
            embeddings_list = embeddings.embed_documents(encoded_images)
        else:
            raise ValueError("Embedding model does not support image embeddings.")

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=encoded_images, 
            metadatas=metadatas
        )
        response["result"] = f"Inserted {len(ids)} images."

    def handle_similarity_search_by_image(self, content: Dict, embeddings, response: Dict):
        """Handle the similarity_search_by_image operation."""
        uri = content.get("uri")
        k = content.get("k", 4)
        filter = content.get("filter")
        if uri is None:
            raise ValueError("Content must include 'uri' for similarity_search_by_image operation")

        encoded_image = self.encode_image(uri)

        if hasattr(embeddings, "embed_documents"):
            image_embedding = embeddings.embed_documents([encoded_image])[0]
        else:
            raise ValueError("Embedding model does not support image embeddings.")

        results = self.collection.query(
            query_embeddings=[image_embedding],
            n_results=k,
            where=filter,
            include=["documents", "metadatas", "distances"]
        )

        parsed_results = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            parsed_results.append({
                'metadata': metadata,
                'distance': distance
            })
        response["result"] = parsed_results

    def handle_similarity_search_by_image_with_relevance_score(self, content: Dict, embeddings, response: Dict):
        """Handle the similarity_search_by_image_with_relevance_score operation."""
        uri = content.get("uri")
        k = content.get("k", 4)
        score_threshold = content.get("score_threshold", 0.5)
        filter = content.get("filter")
        if uri is None:
            raise ValueError("Content must include 'uri' for similarity_search_by_image_with_relevance_score operation")

        # Encode image to base64
        encoded_image = self.encode_image(uri)

        # Embed image
        if hasattr(embeddings, "embed_documents"):
            image_embedding = embeddings.embed_documents([encoded_image])[0]
        else:
            raise ValueError("Embedding model does not support image embeddings.")

        results = self.collection.query(
            query_embeddings=[image_embedding],
            n_results=k,
            where=filter,
            include=["documents", "metadatas", "distances"]
        )

        relevance_scores = []
        for distance in results['distances'][0]:

            relevance_score = 1.0 - distance  
            relevance_scores.append(relevance_score)

        parsed_results = []
        for doc, metadata, score in zip(
            results['documents'][0],
            results['metadatas'][0],
            relevance_scores
        ):
            if score >= score_threshold:
                parsed_results.append({
                    'document': doc,  
                    'metadata': metadata,
                    'relevance_score': score
                })
        response["result"] = parsed_results


    @staticmethod
    def maximal_marginal_relevance(
        query_embedding: np.ndarray,
        embedding_list: np.ndarray,
        k: int = 4,
        lambda_mult: float = 0.5,
    ) -> List[int]:
        if min(k, len(embedding_list)) <= 0:
            return []
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        similarity_to_query = ChromaDBFlow.cosine_similarity(query_embedding, embedding_list)[0]
        most_similar = int(np.argmax(similarity_to_query))
        idxs = [most_similar]
        selected = np.array([embedding_list[most_similar]])
        while len(idxs) < min(k, len(embedding_list)):
            best_score = -np.inf
            idx_to_add = -1
            similarity_to_selected = ChromaDBFlow.cosine_similarity(embedding_list, selected)
            for i, query_score in enumerate(similarity_to_query):
                if i in idxs:
                    continue
                redundant_score = max(similarity_to_selected[i])
                equation_score = (
                    lambda_mult * query_score - (1 - lambda_mult) * redundant_score
                )
                if equation_score > best_score:
                    best_score = equation_score
                    idx_to_add = i
            if idx_to_add == -1:
                break  
            idxs.append(idx_to_add)
            selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
        return idxs

    @staticmethod
    def cosine_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        
        if len(X) == 0 or len(Y) == 0:
            return np.array([])

        X_norm = np.linalg.norm(X, axis=1, keepdims=True)
        Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
        similarity = np.dot(X, Y.T) / (X_norm * Y_norm.T)
        similarity[np.isnan(similarity)] = 0.0
        return similarity


    def encode_image(self, uri: str) -> str:
        """Get base64 string from image URI."""
        if not os.path.isfile(uri):
            raise ValueError(f"Image file '{uri}' does not exist.")
        with open(uri, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def __del__(self):
        """Clean up resources when the object is deleted."""
        try:
            if hasattr(self, "client"):
                del self.client 
                self.logger.info("ChromaDB client deleted.")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

