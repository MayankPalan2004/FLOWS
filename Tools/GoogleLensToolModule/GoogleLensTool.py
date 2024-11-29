from typing import Dict, Any, Optional
import requests
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage

class GoogleLensTool(AtomicFlow):
    """
    A tool to perform Google Lens-like image search using SerpApi's Google Lens API.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.flow_config.get("API_KEY")
        self.base_url = "https://serpapi.com/search"

        if not self.api_key:
            raise ValueError("API_KEY is required for accessing SerpApi.")

    def run(self, input_message: FlowMessage):
        input_data = input_message.data
        image_url = input_data.get("image_url")

        if not image_url:
            response = {'error': 'Image URL cannot be empty'}
        else:
            response = self.perform_image_search(image_url)

        reply = self.package_output_message(
            input_message=input_message,
            response=response,
        )
        self.send_message(reply)

    def perform_image_search(self, image_url: str) -> Dict[str, Any]:
        params = {
            "engine": "google_lens",
            "api_key": self.api_key,
            "url": image_url,
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            result = response.json()
            return self.format_results(result)
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}

    def format_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if result.get("search_metadata", {}).get("status") != "Success":
            return {"error": "Google Lens search failed"}

        formatted_results = {}
        xs = ""  

        if "knowledge_graph" in result:
            subject = result["knowledge_graph"]
            xs += f"Subject: {subject.get('title')} ({subject.get('subtitle', '')})\n"
            xs += f"Link to subject: {subject.get('source_url')}\n\n"

        xs += "Related Images:\n\n"
        if "visual_matches" in result:
            for image in result["visual_matches"]:
                xs += f"Title: {image.get('title')}\n"
                xs += f"Source({image.get('source')}): {image.get('link')}\n"
                xs += f"Image: {image.get('thumbnail')}\n\n"

        if "reverse_image_search" in result:
            xs += f"Reverse Image Search Link: {result['reverse_image_search'].get('link')}\n"

        return {"status": "success", "data": xs}
