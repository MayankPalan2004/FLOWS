import os
import requests
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage

class ZenGuardTool(AtomicFlow):
    """
    A tool to check AI prompts for various security vulnerabilities using the ZenGuard AI API.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.flow_config.get("API_KEY")  
        if not self.api_key:
            raise ValueError("API_KEY is required in the configuration.")
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        self.endpoints = {
            "prompt_injection": "https://api.zenguard.ai/v1/detect/prompt_injection",
            "pii": "https://api.zenguard.ai/v1/detect/pii",
            "allowed_topics": "https://api.zenguard.ai/v1/detect/allowed_topics",
            "banned_topics": "https://api.zenguard.ai/v1/detect/banned_topics",
            "keywords": "https://api.zenguard.ai/v1/detect/keywords",
            "secrets": "https://api.zenguard.ai/v1/detect/secrets",
            "toxicity": "https://api.zenguard.ai/v1/detect/toxicity",
        }

    def run(self, input_message: FlowMessage):
        input_data = input_message.data
        prompt = input_data.get("prompt")
        detectors = input_data.get("detectors", ["prompt_injection"]) 

        if not prompt:
            response = {'error': 'Prompt cannot be empty'}
        else:
            response = self.check_security_issues(prompt, detectors)

        reply = self.package_output_message(
            input_message=input_message,
            response=response,
        )
        self.send_message(reply)

    def check_security_issues(self, prompt, detectors):
        results = {}
        for detector in detectors:
            endpoint = self.endpoints.get(detector)
            if not endpoint:
                results[detector] = {"error": f"Invalid detector: {detector}"}
                continue

         
            data = {"messages": [prompt]}

            try:
                response = requests.post(endpoint, json=data, headers=self.headers)
                if response.status_code == 200:
                    result = response.json()
                else:
                    result = {
                        "error": f"API request failed with status code {response.status_code}",
                        "detail": response.json()
                    }
                results[detector] = result
            except requests.exceptions.RequestException as e:
                results[detector] = {"error": f"API request failed: {str(e)}"}

        return results
