
from copy import deepcopy
from typing import Dict, Any, Optional
import hydra
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage
from aiflows.prompt_template import JinjaPrompt
from aiflows.backends.llm_lite import LiteLLMBackend

class ShortTermMemoryFlow(AtomicFlow):
    """
    A flow that maintains short-term conversational memory.
    
    Operations:
        - update: Integrates new content into the running memory context.
        - fetch: Retrieves relevant content from memory based on a query.

    """

    REQUIRED_KEYS_CONFIG = ["backend"]

    SUPPORTS_CACHING: bool = True

    def __init__(
        self,
        backend: LiteLLMBackend,
        update_prompt_template: JinjaPrompt,
        fetch_prompt_template: JinjaPrompt,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backend = backend
        self.update_prompt_template = update_prompt_template
        self.fetch_prompt_template = fetch_prompt_template
        self.set_up_flow_state()

    def set_up_flow_state(self):
        super().set_up_flow_state()
        self.flow_state["running_context"] = []  

    @classmethod
    def instantiate_from_config(cls, config: Dict) -> 'ShortTermMemoryFlow':
        flow_config = deepcopy(config)

        kwargs = {"flow_config": flow_config}
        kwargs["backend"] = hydra.utils.instantiate(flow_config["backend"], _convert_="partial")
        kwargs["update_prompt_template"] = hydra.utils.instantiate(
            flow_config["update_prompt_template"], _convert_="partial"
        )
        kwargs["fetch_prompt_template"] = hydra.utils.instantiate(
            flow_config["fetch_prompt_template"], _convert_="partial"
        )
        
        return cls(**kwargs)

    def run(self, input_message: FlowMessage):
        """Run the flow based on the specified operation in `input_message`."""
        input_data = input_message.data
        operation = input_data.get("operation")
        input_data =input_data.get("content")

        if operation == "update":
            content = input_data.get("content")
            result = self.handle_update(content)
        elif operation == "fetch":
            query = input_data.get("query")
            participant = input_data.get("participant")
            result = self.handle_fetch(query, participant)
        else:
            result = {"error": f"Unsupported operation '{operation}' in ShortTermMemoryFlow."}
        
        reply_message = self.package_output_message(
            input_message=input_message,
            response=result
        )
        self.send_message(reply_message)



    def handle_update(self, content: str) -> Dict[str, Any]:
        """Handles memory update by combining new content into the running context."""
        update_input = {
            "running_context": self.flow_state["running_context"][-1] if self.flow_state["running_context"] else "",
            "new_message": content
        }
        prompt_text = self.update_prompt_template.format(**update_input)

        response = self.backend(messages=[{"role": "system", "content": prompt_text}])
        updated_memory = response[0]["content"] if response else ""



        if "Memory Entries:" in updated_memory:
            extracted_memory = updated_memory.split("Memory Entries:")[-1].strip()
        else:
            extracted_memory = updated_memory.strip()

        self.flow_state["running_context"] = [extracted_memory]
        return {"result": "Memory updated successfully."}


    def handle_fetch(self, query: str, participant: str) -> Dict[str, Any]:
        """Handles memory retrieval by fetching relevant content from the running context."""
        fetch_input = {
            "running_context": self.flow_state["running_context"][-1] if self.flow_state["running_context"] else "",
            "query": query,
            "participant": participant
        }

     
        prompt_text = self.fetch_prompt_template.format(**fetch_input)

        response = self.backend(messages=[{"role": "system", "content": prompt_text}])
        retrieved_memory = response[0]["content"] if response else ""

        if "Relevant Memories:" in retrieved_memory:
            cleaned_memory = retrieved_memory.split("Relevant Memories:")[-1].strip()
        else:
            cleaned_memory = retrieved_memory.strip()
            
        return {"result": cleaned_memory}


