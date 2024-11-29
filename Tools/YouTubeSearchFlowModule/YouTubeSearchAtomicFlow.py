from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage
import requests

class YouTubeSearchAtomicFlow(AtomicFlow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.flow_config.get("API_KEY")  
        self.base_url = "https://www.googleapis.com/youtube/v3/search"  

        self.default_options = {
            "part": "snippet",           
            "type": "video",             
            "maxResults": 5,             
            "order": "relevance",        
            "videoDuration": "any"      
        }

    def run(self, input_message: FlowMessage):
        input_data = input_message.data
        query = input_data.get("query")
        options = input_data.get("options", {})  

        if not query:
            response = {'error': 'No query provided'}
        else:
            response = self.perform_search(query, options)

        reply = self.package_output_message(
            input_message=input_message,
            response=response,
        )
        self.send_message(reply)

    def perform_search(self, query, options):
        try:
            search_options = {**self.default_options, **options, "q": query, "key": self.api_key}

            response = requests.get(self.base_url, params=search_options)

            if response.status_code == 200:
                results = response.json()
                formatted_results = self.format_results(results)
                return {'results': formatted_results}
            else:
                return {'error': f"API request failed with status code {response.status_code}"}
        except Exception as e:
            return {'error': str(e)}

    def format_results(self, results):
        formatted_results = []
        for item in results.get("items", []):
            if item["id"]["kind"] == "youtube#video":
                formatted_results.append({
                    "Title": item["snippet"]["title"],
                    "Description": item["snippet"]["description"],
                    "Video URL": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                    "Channel Title": item["snippet"]["channelTitle"],
                    "Published At": item["snippet"]["publishedAt"]
                })
        return formatted_results
