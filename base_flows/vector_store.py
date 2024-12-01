# vector_store.py

import chromadb
from chromadb.config import Settings
from datetime import datetime
import uuid
import hashlib
from typing import List, Dict, Any
from langchain_nomic import NomicEmbeddings

from aiflows.backends.api_info import ApiInfo

class OpenAIEmbeddingFunction:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def __call__(self, input):
        return self.embedding_model.embed_documents(input)

class VectorStore:
    def __init__(self, collection_name: str, persist_directory: str, vector_backend):
        self.vector_backend = vector_backend
        self.embeddings_model = self.get_embeddings_model()
        settings = Settings(
            anonymized_telemetry=False,
            persist_directory=persist_directory,
            is_persistent=True
        )
        self.client = chromadb.Client(settings)
        self.collection_name = collection_name
        embedding_function = OpenAIEmbeddingFunction(self.embeddings_model)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_function
        )

    def get_embeddings_model(self):
        api_information = self.vector_backend.get_key()
        if api_information.backend_used == "cohere":
           embeddings = NomicEmbeddings(model="nomic-embed-text-v1")
        else:
            embeddings = NomicEmbeddings(model="nomic-embed-text-v1")
        return embeddings

    def add_memory(self, text: str, metadata: Dict[str, Any] = None):
        """
        Adds a new memory to the vector store with optional metadata.
        """
        memory_id = str(uuid.uuid4())
        payload = {
            "id": memory_id,  # Store the ID in metadata
            "created_at": datetime.utcnow().isoformat(),
            "hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
        }
        # Merge additional metadata if provided
        if metadata:
            payload.update(metadata)
        embeddings = self.embeddings_model.embed_documents([text])
        self.collection.add(
            ids=[memory_id],
            embeddings=embeddings,
            documents=[text],
            metadatas=[payload],
        )



    def update_memory_entry(self, memory_id: str, text: str, metadata: Dict[str, Any] = None):
        """
        Updates an existing memory entry.

        Args:
            memory_id (str): The ID of the memory to update.
            text (str): The new text for the memory.
            metadata (Dict[str, Any], optional): Updated metadata, such as user_id.
        """
        payload = {
            "id": memory_id,  
            "updated_at": datetime.utcnow().isoformat(),
            "hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
        }
        # Merge additional metadata if provided
        if metadata:
            payload.update(metadata)
        embeddings = self.embeddings_model.embed_documents([text])
        self.collection.update(
            ids=[memory_id],
            embeddings=embeddings,
            documents=[text],
            metadatas=[payload],
        )


    def delete_memory(self, memory_id: str,metadata):
        self.collection.delete(ids=[memory_id])

    def retrieve_existing_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all existing memories for a specific user.
        """
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=["documents", "metadatas"],
            )
            if not results or not results.get("documents"):
                return []

            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            # Extract IDs from metadatas
            ids = [metadata.get("id") for metadata in metadatas]

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
            print(f"Error retrieving existing memories: {e}")
            return []


    def search_memories(self, query: str, user_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for memories matching the query and user_id.
        """
        embeddings = self.embeddings_model.embed_documents([query])
        results = self.collection.query(
            query_embeddings=embeddings,
            n_results=n_results,
            where={"user_id": user_id},  # Filter by user_id
            include=["documents", "metadatas"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        # Extract IDs from metadatas
        ids = [metadata.get("id") for metadata in metadatas]

        combined_results = [
            {
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i] if i < len(metadatas) else {}
            }
            for i in range(len(documents))
        ]
        return combined_results

