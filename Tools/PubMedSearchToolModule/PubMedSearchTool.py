import requests
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage

class PubMedSearchTool(AtomicFlow):
    """
    A tool to perform searches on PubMed for biomedical articles.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.base_url_summary = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

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
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": options.get("limit", 5), 
            "retmode": "json",
            "mindate": options.get("year_from"),
            "maxdate": options.get("year_to"),
            "sort": options.get("sort", "relevance")
        }

        try:
            search_response = requests.get(self.base_url_search, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()
            ids = search_data.get("esearchresult", {}).get("idlist", [])

            if ids:
                return self.get_article_summaries(ids)
            else:
                return {"status": "success", "data": []}
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}

    def get_article_summaries(self, ids):
        summary_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json"
        }

        try:
            summary_response = requests.get(self.base_url_summary, params=summary_params)
            summary_response.raise_for_status()
            summary_data = summary_response.json()

            results = []
            for uid, item in summary_data.get("result", {}).items():
                if uid == "uids":
                    continue
                results.append({
                    "title": item.get("title"),
                    "authors": [author.get("name") for author in item.get("authors", [])],
                    "source": item.get("source"),
                    "pubdate": item.get("pubdate"),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
                })
            return {"status": "success", "data": results}
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
