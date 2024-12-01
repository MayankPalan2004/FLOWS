
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import uuid
import logging
from pydantic import BaseModel
import hydra
from copy import deepcopy
from langchain.document_loaders import TextLoader
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
)

from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings

class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[Dict]
    vector: Optional[List[float]]
    document: Optional[str]  

class QdrantDBFlow(AtomicFlow):
    """
    An atomic flow that interfaces with a Qdrant vector store for vector operations.

    Supports operations:
    - insert
    - search
    - update
    - delete
    - get
    - list_collections
    - delete_collection
    - collection_info
    - list_vectors
    """

    def __init__(self, backend, **kwargs):
        super().__init__(**kwargs)
        self.backend = backend 
        self.logger = logging.getLogger(__name__)
        self.flow_state["db_initialized"] = False

    def set_up_flow_state(self):
        super().set_up_flow_state()
        self.flow_state["db_initialized"] = False

    @classmethod
    def _set_up_backend(cls, config):
        """This instantiates the backend of the flow from a configuration file.

        :param config: The configuration of the backend.
        :type config: Dict[str, Any]
        :return: The backend of the flow.
        :rtype: Dict[str, Any]
        """
        kwargs = {}
        kwargs["backend"] = hydra.utils.instantiate(config['backend'], _convert_="partial")
        return kwargs

    @classmethod
    def instantiate_from_config(cls, config):
        """This method instantiates the flow from a configuration file.

        :param config: The configuration of the flow.
        :type config: Dict[str, Any]
        :return: The instantiated flow.
        :rtype: QdrantDBFlow
        """
        flow_config = deepcopy(config)
        kwargs = {"flow_config": flow_config}

        kwargs.update(cls._set_up_backend(flow_config))
        return cls(**kwargs)

    def initialize_db(self):
        
            self.backend_config = self.flow_config.get('backend', {})
            self.collection_name = self.flow_config.get('collection_name', 'default_collection')
            self.embedding_model_dims = self.flow_config.get('embedding_model_dims', 768)  

            self.paths_to_data = self.flow_config.get('paths_to_data', [])
            self.chunk_size = self.flow_config.get('chunk_size', 1000)
            self.chunk_overlap = self.flow_config.get('chunk_overlap', 0)
            self.separator = self.flow_config.get('separator', '\n\n')

            self.initialize_client()

            self.create_collection(on_disk=self.backend_config.get('on_disk', False))
            self.flow_state["db_initialized"] = True

            if self.paths_to_data:
                self.load_initial_data()
    def get_embeddings_model(self):
        api_information = self.backend.get_key()
        if api_information.backend_used == "cohere":
            embeddings = CohereEmbeddings(model="embed-english-v3.0")
        else:
            embeddings = CohereEmbeddings(model="embed-english-v3.0")
        return embeddings

    def initialize_client(self):
        client_params = {}
        api_key = self.backend_config.get('api_key')
        url = self.backend_config.get('url')
        host = self.backend_config.get('host')
        port = self.backend_config.get('port')
        path = self.backend_config.get('path')

        if api_key:
            client_params['api_key'] = api_key
        if url:
            client_params['url'] = url
        elif host and port:
            client_params['host'] = host
            client_params['port'] = port
        elif path:
            client_params['path'] = path
        else:
            raise ValueError("No valid connection parameters provided for Qdrant client.")

        self.client = QdrantClient(**client_params)

    def create_collection(self, on_disk: bool):
        collections = self.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_model_dims,
                    distance=Distance.COSINE,
                    on_disk=on_disk
                )
            )
            self.logger.info(f"Created collection '{self.collection_name}'")
        else:
            self.logger.info(f"Collection '{self.collection_name}' already exists")

    

    def load_initial_data(self):
        self.logger.info("Loading data from provided paths...")
        embeddings_model = self.get_embeddings_model()
        full_docs = []
        metadatas = []
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=self.separator
        )
        for path in self.paths_to_data:
            loader = TextLoader(path)
            documents = loader.load()
            docs = text_splitter.split_documents(documents)
            full_docs.extend(docs)
            for doc in docs:
                metadata = {
                    'source': path,
                    'hash': hashlib.md5(doc.page_content.encode('utf-8')).hexdigest(),
                    'created_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat(),
                    'document': doc.page_content
                }
                metadatas.append(metadata)
        texts = [doc.page_content for doc in full_docs]
        embeddings_list = embeddings_model.embed_documents(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        points = [
            PointStruct(
                id=ids[i],
                vector=embeddings_list[i],
                payload=metadatas[i]
            )
            for i in range(len(ids))
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        self.logger.info(f"Loaded {len(texts)} documents into the collection.")

    def run(self, input_message: FlowMessage):
        """Run the flow, supporting various operations with Qdrant."""
        self.initialize_db()
        input_data = input_message.data
        embeddings_model = self.get_embeddings_model()
        response = {}

        operation = input_data.get("operation")
        content = input_data.get("content", {})

        if operation not in [
            "insert", "search", "update", "delete", "get",
            "list_collections", "delete_collection", "collection_info", "list_vectors"
        ]:
            raise ValueError(f"Operation '{operation}' not supported")

        if operation == "insert":
            self.handle_insert(content, embeddings_model, response)
        elif operation == "search":
            self.handle_search(content, embeddings_model, response)
        elif operation == "update":
            self.handle_update(content, embeddings_model, response)
        elif operation == "delete":
            self.handle_delete(content, response)
        elif operation == "get":
            self.handle_get(content, response)
        elif operation == "list_collections":
            self.handle_list_collections(response)
        elif operation == "delete_collection":
            self.handle_delete_collection(response)
        elif operation == "collection_info":
            self.handle_collection_info(response)
        elif operation == "list_vectors":
            self.handle_list_vectors(content, response)
        else:
            raise ValueError(f"Unknown operation '{operation}'")

        reply = self.package_output_message(
            input_message=input_message,
            response=response
        )
        self.send_message(reply)

    def handle_insert(self, content, embeddings_model, response):
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
            payload['document'] = vectors[i]  

        embeddings_list = embeddings_model.embed_documents(vectors)

        points = [
            PointStruct(
                id=ids[i],
                vector=embeddings_list[i],
                payload=payloads[i]
            )
            for i in range(len(ids))
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        response["result"] = f"Inserted {len(ids)} vectors."

    def _create_filter(self, filters: dict) -> Optional[Filter]:
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict) and ("gte" in value or "lte" in value):
                conditions.append(FieldCondition(key=key, range=Range(**value)))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions) if conditions else None

    def handle_search(self, content, embeddings_model, response):
        query = content.get("query")
        limit = content.get("limit", 5)
        filters = content.get("filters")
        if query is None:
            raise ValueError("Content must include 'query' for search operation")

        if not isinstance(query, list):
            query = [query]

        query_embeddings = embeddings_model.embed_documents(query)

        query_filter = self._create_filter(filters)

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embeddings[0],
            query_filter=query_filter,
            limit=limit,
            with_vectors=False,
            with_payload=True
        )

        parsed_results = []
        for hit in hits:
            parsed_results.append({
                'id': str(hit.id),
                'score': hit.score,
                'payload': hit.payload,
                'vector': None, 
                'document': hit.payload.get('document', None)
            })

        response["result"] = parsed_results

    def handle_update(self, content, embeddings_model, response):
        vector_id = content.get("id")
        vector = content.get("vector")
        payload = content.get("payload")
        if vector_id is None:
            raise ValueError("Content must include 'id' for update operation")

        embeddings_vector = None
        if vector is not None:
            embeddings_vector = embeddings_model.embed_documents([vector])[0]

        if payload is None:
            payload = {}
        payload['updated_at'] = datetime.utcnow().isoformat()
        if vector:
            payload['document'] = vector

        point = PointStruct(
            id=vector_id,
            vector=embeddings_vector,
            payload=payload
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        response["result"] = f"Updated vector with id {vector_id}."

    def handle_delete(self, content, response):
        vector_id = content.get("id")
        if vector_id is None:
            raise ValueError("Content must include 'id' for delete operation")

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(
                points=[vector_id],
            ),
        )
        response["result"] = f"Deleted vector with id {vector_id}."

    def handle_get(self, content, response):
        vector_id = content.get("id")
        if vector_id is None:
            raise ValueError("Content must include 'id' for get operation")
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[vector_id],
            with_payload=True,
            with_vectors=True
        )
        if result:
            point = result[0]
            response["result"] = {
                'id': str(point.id),
                'payload': point.payload,
                'vector': point.vector,
                'document': point.payload.get('document', None)
            }
        else:
            response["result"] = f"No vector found with id {vector_id}."

    def handle_list_collections(self, response):
        collections = self.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        response["result"] = collection_names

    def handle_delete_collection(self, response):
        self.client.delete_collection(
            collection_name=self.collection_name
        )
        response["result"] = f"Deleted collection '{self.collection_name}'."

    def handle_collection_info(self, response):
        info = self.client.get_collection(
            collection_name=self.collection_name
        )
        response["result"] = {
            "status": info.status,
            "vectors_count": info.vectors_count,
            "config": info.config.dict()
        }

    def handle_list_vectors(self, content, response):
        filters = content.get("filters")
        limit = content.get("limit", 100)
        query_filter = self._create_filter(filters)
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=True
        )
        points, _ = scroll_result
        parsed_results = []
        for point in points:
            parsed_results.append({
                'id': str(point.id),
                'payload': point.payload,
                'vector': point.vector,
                'document': point.payload.get('document', None)
            })
        response["result"] = parsed_results
