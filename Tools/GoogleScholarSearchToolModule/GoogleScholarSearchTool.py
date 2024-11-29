import requests
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage

class GoogleScholarSearchTool(AtomicFlow):
    """
    A tool to perform searches on Google Scholar for academic papers.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.flow_config.get("API_KEY")
        self.base_url = "https://serpapi.com/search"

        if not self.api_key:
            raise ValueError("API_KEY is required for accessing SerpAPI.")

    def run(self, input_message: FlowMessage):
        input_data = input_message.data
        query = input_data.get("query")
        options = input_data.get("options", {})

        if not query:
            response = {'error': 'Query cannot be empty'}
        else:
            response = self.perform_search(query, options)

        reply = self.package_output_message(
            input_message=input_message,
            response=response,
        )
        self.send_message(reply)

    def perform_search(self, query, options):
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self.api_key,
            "num": options.get("num", 5), 
            "as_ylo": options.get("year_from"), 
            "as_yhi": options.get("year_to"),    
            "scisbd": options.get("sort", "0"),  
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            result = response.json()
            return self.format_results(result)
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}

    def format_results(self, result):
        results = []
        for item in result.get("organic_results", []):
            formatted_item = {
                "title": item.get("title"),
                "author": item.get("publication_info", {}).get("authors"),
                "year": item.get("publication_info", {}).get("year"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            }
            results.append(formatted_item)
        return {"status": "success", "data": results}
