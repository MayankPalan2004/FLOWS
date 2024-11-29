import requests
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage

class WikipediaSearchTool(AtomicFlow):
    """
    A tool to perform searches on Wikipedia for summaries or full article content.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://en.wikipedia.org/w/api.php"

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
        if options.get("summary", True):
            return self.get_summary(query, options)
        else:
            return self.get_full_content(query, options)

    def get_summary(self, query, options):
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,  
            "explaintext": True,  
            "titles": query,
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})

        results = [
            {
                "title": page.get("title"),
                "description": page.get("extract"),
                "url": f"https://en.wikipedia.org/wiki/{page.get('title').replace(' ', '_')}"
            }
            for page in pages.values()
        ]
        return {"status": "success", "data": results}

    def get_full_content(self, query, options):
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "titles": query,
            "format": "json",
            "exintro": options.get("intro", False)
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})

        results = [
            {"title": page.get("title"), "content": page.get("extract")}
            for page in pages.values()
        ]
        return {"status": "success", "data": results}
