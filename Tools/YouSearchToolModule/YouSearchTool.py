import os
import requests
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage

class YouSearchTool(AtomicFlow):
    """
    A tool to perform searches on You.com for various content types (web, image, news).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.flow_config.get("API_KEY")  # Access API key from configuration
        if not self.api_key:
            raise ValueError("API_KEY is required in the configuration.")
        self.endpoints = {
            "web": "https://api.you.com/v1/search",
            "image": "https://api.you.com/v1/image",
            "news": "https://api.you.com/v1/news",
        }

    def run(self, input_message: FlowMessage):
        input_data = input_message.data
        query = input_data.get("query")
        search_type = input_data.get("search_type", "web")  # Default to web search
        options = input_data.get("options", {})  # Additional search options

        if not query:
            response = {'error': 'Query cannot be empty'}
        else:
            response = self.perform_search(query, search_type, options)

        reply = self.package_output_message(
            input_message=input_message,
            response=response,
        )
        self.send_message(reply)

    def perform_search(self, query, search_type, options):
        endpoint = self.endpoints.get(search_type)
        if not endpoint:
            return {'error': f"Invalid search type: {search_type}"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        params = {"q": query, **options}


        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            result = response.json()
            return {"status": "success", "data": result}
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
