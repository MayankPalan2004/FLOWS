from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage
import requests

class BingSearchAtomicFlow(AtomicFlow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.flow_config.get("API_KEY")  
        self.base_url = "https://api.bing.microsoft.com/v7.0"  

        self.default_options = {
            "web": {
                "mkt": "en-US",
                "safesearch": "Moderate",
                "count": 10,
                "offset": 0,
                "responseFilter": "Webpages"
            },
            "image": {
                "mkt": "en-US",
                "safesearch": "Moderate",
                "count": 10,
                "offset": 0,
                "imageType": "Photo",
                "size": "Medium",
                "aspect": "All"
            },
            "video": {
                "mkt": "en-US",
                "safesearch": "Moderate",
                "count": 10,
                "offset": 0,
                "videoLength": "All",
                "videoType": "All"
            },
            "news": {
                "mkt": "en-US",
                "safesearch": "Moderate",
                "count": 10,
                "offset": 0,
                "category": "Business"
            }
        }

    def run(self, input_message: FlowMessage):
        input_data = input_message.data
        query = input_data.get("query")
        search_type = input_data.get("search_type", "web")  
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
            headers = {"Ocp-Apim-Subscription-Key": self.api_key}
            params = {"q": query, **search_options}

            endpoint = f"{self.base_url}/{search_type}s/search"

            response = requests.get(endpoint, headers=headers, params=params)

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
        if search_type == "web":
            for result in results.get("webPages", {}).get("value", []):
                formatted_results.append({
                    "Name": result.get("name"),
                    "URL": result.get("url"),
                    "Snippet": result.get("snippet"),
                })
        elif search_type == "image":
            for result in results.get("value", []):
                formatted_results.append({
                    "Name": result.get("name"),
                    "Thumbnail URL": result.get("thumbnailUrl"),
                    "Content URL": result.get("contentUrl"),
                    "Host Page URL": result.get("hostPageUrl"),
                })
        elif search_type == "video":
            for result in results.get("value", []):
                formatted_results.append({
                    "Name": result.get("name"),
                    "Thumbnail URL": result.get("thumbnailUrl"),
                    "Content URL": result.get("contentUrl"),
                    "Host Page URL": result.get("hostPageUrl"),
                    "Duration": result.get("duration"),
                })
        elif search_type == "news":
            for result in results.get("value", []):
                formatted_results.append({
                    "Name": result.get("name"),
                    "URL": result.get("url"),
                    "Description": result.get("description"),
                    "Date Published": result.get("datePublished"),
                    "Provider": result.get("provider")[0].get("name") if result.get("provider") else None,
                })
        return formatted_results
