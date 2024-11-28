
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage
import requests

class TavilySearchAtomicFlow(AtomicFlow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.flow_config.get("API_KEY")  
        self.base_url = "https://api.tavily.com/search" 

        self.default_options = {
            "text": {
                "search_depth": "basic",
                "topic": "general",
                "max_results": 5,
                "include_answer": False,
                "include_images": False,
                "include_image_descriptions": False,
                "include_raw_content": False,
                "include_domains": [],
                "exclude_domains": []
            },
            "images": {
                "search_depth": "basic",
                "topic": "general",
                "max_results": 5,
                "include_images": True,
                "include_image_descriptions": True
            },
            "videos": {
                "search_depth": "basic",
                "topic": "general",
                "max_results": 5
            },
            "news": {
                "search_depth": "basic",
                "topic": "news",
                "days": 3,
                "max_results": 5
            },
            "maps": {
                "search_depth": "basic",
                "topic": "general",
                "max_results": 5
            }
        }

    def run(self, input_message: FlowMessage):
        input_data = input_message.data
        query = input_data.get("query")
        search_type = input_data.get("search_type", "text")  
        options = input_data.get("options", {})  

        if not query:
            response = {'error': 'No query provided'}
        else:
            response = self.perform_search(query, search_type, options)

        reply = self.package_output_message(
            input_message=input_message,
            response=response,
        )
        self.send_message(reply)

    def perform_search(self, query, search_type, options):
        try:
            search_options = {**self.default_options.get(search_type, {}), **options}

            payload = {
                "api_key": self.api_key,
                "query": query,
                **search_options
            }

            response = requests.post(self.base_url, json=payload)

            if response.status_code == 200:
                results = response.json()
                formatted_results = self.format_results(results, search_type)
                return {'results': formatted_results}
            else:
                return {'error': f"API request failed with status code {response.status_code}"}
        except Exception as e:
            return {'error': str(e)}

    def format_results(self, results, search_type):
        formatted_results = []
        for result in results.get("results", []):
            if search_type == "text":
                formatted_results.append({
                    "Title": result.get("title"),
                    "URL": result.get("url"),
                    "Snippet": result.get("content"),
                })
            elif search_type == "images":
                formatted_results.append({
                    "Title": result.get("title"),
                    "Image URL": result.get("url"),
                    "Description": result.get("description"),
                })
            elif search_type == "videos":
                formatted_results.append({
                    "Title": result.get("title"),
                    "Video URL": result.get("url"),
                    "Description": result.get("content"),
                })
            elif search_type == "news":
                formatted_results.append({
                    "Title": result.get("title"),
                    "URL": result.get("url"),
                    "Published Date": result.get("published_date"),
                    "Source": result.get("source"),
                })
            elif search_type == "maps":
                formatted_results.append({
                    "Name": result.get("name"),
                    "Address": result.get("address"),
                    "Latitude": result.get("latitude"),
                    "Longitude": result.get("longitude"),
                })
        return formatted_results
