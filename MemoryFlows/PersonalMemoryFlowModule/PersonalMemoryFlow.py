
import os
import uuid
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any
from copy import deepcopy
import pytz
import hydra

from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage
from aiflows.prompt_template import JinjaPrompt
from langchain_nomic import NomicEmbeddings


import chromadb
from chromadb.config import Settings
class OpenAIEmbeddingFunction:
    def __init__(self, openai_embedding_model):
        self.embedding_model = openai_embedding_model

    def __call__(self, input):
        return self.embedding_model.embed_documents(input)

class PersonalMemoryFlow(AtomicFlow):
    """
    PersonalMemoryFlow for managing memory updates and retrievals:
    1. Update: Extract facts using FACT_RETRIEVAL_PROMPT and decide memory actions (add/update/delete) with UPDATE_MEMORY_PROMPT_TEMPLATE.
    2. Fetch: Retrieve memories relevant to a query.
    """

    def __init__(
        self,
        llm_backend,
        vector_backend,
        update_prompt_template: JinjaPrompt,
        fact_retrieval_prompt_template: JinjaPrompt,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.llm_backend = llm_backend
        self.vector_backend = vector_backend
        self.update_prompt_template = update_prompt_template
        self.fact_retrieval_prompt_template = fact_retrieval_prompt_template
        self.embeddings_model = NomicEmbeddings(model="nomic-embed-text-v1")
        self.initialize_collection()
        self.embeddings = self.get_embeddings_model()

        

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

        return cls(**kwargs)


    def initialize_collection(self):
        """
        Initialize the ChromaDB collection.
        If no data exists, prepare an empty collection for future usage.
        """
        settings = Settings(
            anonymized_telemetry=False,
            persist_directory=self.flow_config.get("persist_directory", "./chroma_db123_dir"),
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
            print(f"Collection '{self.collection_name}' is empty. Ready for new memories.")
        else:
            print(f"Collection '{self.collection_name}' contains {collection_count} memories.")

        self.flow_state["db_initialized"] = True


    def run(self, input_message: FlowMessage):
        input_data = input_message.data
        operation = input_data.get("operation")
        content = input_data.get("content")

        if operation == "update":
            facts = self.extract_facts(content["content"])
           
            response = self.update_memory(facts)

        elif operation == "fetch":
            response = self.fetch_memory(content["query"])
        else:
            response = {"error": f"Unsupported operation '{operation}'."}

        reply_message = self.package_output_message(
            input_message=input_message,
            response=response,
        )
        self.send_message(reply_message)

    def extract_facts(self, conversation: str) -> List[str]:
        """Extract facts from a conversation using FACT_RETRIEVAL_PROMPT."""

 

        update_input = {
            "conversation": conversation}
      
        
        prompt_text = self.fact_retrieval_prompt_template.format(**update_input)
        response = self.llm_backend(messages=[{"role": "system", "content": prompt_text}])
        
        print("FACTS EXTRACTED : ",response)
        try:
            facts_json = json.loads(response[0]["content"])
            return facts_json.get("facts", [])
        except json.JSONDecodeError:
            print('JSON ERROR')
            return []
        

    
    def update_memory(self, facts: List[str]) -> Dict[str, Any]:
        """
        Update the memory collection based on the extracted facts.
        If no existing memories are present, add new ones directly.
        """
        if not facts:
            print("No new facts extracted. Skipping memory update.")
            return {"result": "No new facts extracted."}

        existing_memories = self.retrieve_existing_memories()
        if not existing_memories:
            print("No existing memories found. Adding facts as new memories.")

        update_input = {
            'retrieved_old_memory_dict': json.dumps(existing_memories),
            'response_content': json.dumps(facts)
        }

        prompt_text = self.update_prompt_template.format(**update_input)
        response = self.llm_backend(messages=[{"role": "system", "content": prompt_text}])
        raw_content = response[0]["content"] if response else ""

        try:
            cleaned_content = raw_content.strip("`").strip()
            print(f"Cleaned LLM Response: {cleaned_content}")
            memory_updates = json.loads(cleaned_content).get("memory", [])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing memory updates: {e}")
            return {"result": "Failed to process memory updates."}

        for action in memory_updates:
            event = action["event"]
            new_text = action["text"]
            old_memory_text = action.get("old_memory", None)

            if event == "ADD":
                self.add_memory(new_text)
            elif event == "UPDATE":
                old_memory_id = next(
                    (mem["id"] for mem in existing_memories if mem["text"] == old_memory_text),
                    None
                )
                if old_memory_id:
                    print(f"Updating memory with ID: {old_memory_id}")
                    self.update_memory_entry(old_memory_id, new_text)
                else:
                    print(f"No matching memory found for update: {old_memory_text}")
            elif event == "DELETE":
                old_memory_id = next(
                    (mem["id"] for mem in existing_memories if mem["text"] == old_memory_text),
                    None
                )
                if old_memory_id:
                    print(f"Deleting memory with ID: {old_memory_id}")
                    self.delete_memory(old_memory_id)
                else:
                    print(f"No matching memory found for deletion: {old_memory_text}")
            elif event == "NONE":
                print(f"No changes for memory: {old_memory_text}")
            else:
                print(f"Unrecognized event type: {event}")

        return {"result": "Memory updated successfully."}

    def update_memory_entry(self, memory_id: str, text: str):
        """
        Update an existing memory in the vector store.
        Replaces the old text with the new text while keeping the same ID.
        """
        print(f"Updating memory ID: {memory_id} with text: {text}")
        payload = {
            "updated_at": datetime.utcnow().isoformat(),
            "hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
        }
        self.collection.update(
            ids=[memory_id],
            embeddings=self.embeddings.embed_documents([text]),
            documents=[text],  
            metadatas=[payload], 
        )



    def retrieve_existing_memories(self) -> List[Dict[str, Any]]:
        """
        Retrieve existing memories from the ChromaDB collection.
        If the collection is empty, return an empty list.
        """
        try:
            results = self.collection.get(
                where={}, 
                include=["documents", "metadatas", "uris"]
            )
            print(f'RETRIEVE:',results)
            
            if not results or not results.get("documents"):
                print("No existing memories found in the collection.")
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
            print(f"Error retrieving existing memories: {e}")
            return []


    def apply_memory_updates(self, memory_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply memory updates (add/update/delete)."""
        for action in memory_updates:
            event = action["event"]
            if event == "ADD":
                self.add_memory(action["text"])
            elif event == "UPDATE":
                print('234')
                print(action["id"])
                self.update_memory_entry(action["id"], action["text"])
            elif event == "DELETE":
                self.delete_memory(action["id"])
        return {"result": "Memory updated successfully."}

    def fetch_memory(self, query: str) -> Dict[str, Any]:
        """Fetch relevant memories based on a query."""
        embeddings = self.embeddings.embed_documents([query])
        results = self.collection.query(
            query_embeddings=embeddings,
            n_results=5,
            include=["documents", "metadatas"],
        )
        if not results or "documents" not in results:
            return {"result": "No matching memories found."}

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        ids = results.get("ids", [])

        return {
            "result": [
                {
                    "id": ids[i],
                    "memory": documents[i],
                    "metadata": metadatas[i] if i < len(metadatas) else {}
                }
                for i in range(len(documents))
            ]
        }


    def add_memory(self, text: str):
        """Add new memory to the vector store."""
        memory_id = str(uuid.uuid4())
        payload = {
            "created_at": datetime.utcnow().isoformat(),
            "hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
        }
        self.collection.add(
            ids=[memory_id],
            embeddings=self.embeddings.embed_documents([text]),
            documents=[text],
            metadatas=[payload],
        )

    

    def delete_memory(self, memory_id: str):
        """Delete memory from the vector store."""
        self.collection.delete(ids=[memory_id])

    def get_embeddings_model(self):
        api_information = self.vector_backend.get_key()
        if api_information.backend_used == "cohere":
            embeddings = NomicEmbeddings(model="nomic-embed-text-v1")
        else:
            embeddings = NomicEmbeddings(model="nomic-embed-text-v1")
        return embeddings


        
