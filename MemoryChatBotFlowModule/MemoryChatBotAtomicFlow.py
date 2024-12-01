
import json
import re
import logging

log = logging.getLogger(__name__)



from copy import deepcopy
from typing import Dict, Any
import hydra
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage

class MemoryChatBotAtomicFlow(AtomicFlow):
    """
    A conversational chatbot flow that leverages short-term memory integrated into the base Flow class.
    """

    REQUIRED_KEYS_CONFIG = ["name", "description", "backend", "prompt_template"]

    def __init__(
        self,
        flow_config: Dict[str, Any],
        backend: Any,  
        prompt_template: Any,  
        **kwargs
    ):
        """
        Initializes the ChatBotFlow.
        """
        super().__init__(flow_config=flow_config, **kwargs)
        self.backend = backend
        self.prompt_template = prompt_template
    

    @classmethod
    def instantiate_from_config(cls, config: Dict) -> 'ChatBotFlow':
   
        flow_config = deepcopy(config)
        backend = hydra.utils.instantiate(
            flow_config["backend"], _convert_="partial"
        )
        prompt_template = hydra.utils.instantiate(
            flow_config["prompt_template"], _convert_="partial"
        )
        return cls(flow_config=flow_config, backend=backend, prompt_template=prompt_template)

    def run(self, input_message: FlowMessage):
        """
        Processes the user message and generates a chatbot response with memory context if enabled.
        """
        input_data = input_message.data
        user_message = input_data.get("message", "")
        memory_context = input_data.get("memory_context", "")

        prompt_input = {
        "memory_context": memory_context,
        "user_message": user_message
    }
        prompt = self.prompt_template.format(**prompt_input)

        response = self.backend(messages=[{"role": "user", "content": prompt}])
        bot_reply = response[0]["content"] if response else "I'm sorry, I didn't understand that."

        self.store_last_answer(bot_reply)


        reply_content = {"reply": bot_reply}
        reply_message = self.package_output_message(
            input_message=input_message,
            response=reply_content
        )
        self.send_message(reply_message)





    


    

    



    
    


