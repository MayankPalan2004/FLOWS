import requests
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage

class StackExchangeSearchTool(AtomicFlow):
    """
    A tool to perform searches on Stack Exchange sites like Stack Overflow.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://api.stackexchange.com/2.3/search/advanced"

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
            "q": query,
            "site": options.get("site", "stackoverflow"), 
            "sort": options.get("sort", "relevance"),     
            "order": options.get("order", "desc"),        
            "pagesize": options.get("limit", 5),           
            "accepted": options.get("accepted"),           
            "tagged": options.get("tagged")             
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
        for item in result.get("items", []):
            formatted_item = {
                "title": item.get("title"),
                "author": item.get("owner", {}).get("display_name"),
                "creation_date": item.get("creation_date"),
                "link": item.get("link"),
                "score": item.get("score")
            }
            results.append(formatted_item)
        return {"status": "success", "data": results}
