import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage

class WebsiteScraperTool(AtomicFlow):
  

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, input_message: FlowMessage):
        input_data = input_message.data
        website_url = input_data.get("website_url")
        sections = input_data.get("sections", ["p"])

        if not website_url:
            response = {'error': 'Website URL is required'}
        else:
            response = self.scrape_website(website_url, sections)

        reply = self.package_output_message(
            input_message=input_message,
            response=response,
        )
        self.send_message(reply)

    def scrape_website(self, website_url: str, sections: List[str]):
        try:
            page = requests.get(
                website_url,
                timeout=15,
                headers=self.headers
            )
            page.raise_for_status()
            page.encoding = page.apparent_encoding
            parsed = BeautifulSoup(page.text, "html.parser")

            scraped_content = {}
            for section in sections:
                elements = parsed.find_all(section)
                scraped_content[section] = [element.get_text(strip=True) for element in elements]

            if not scraped_content:
                return {"error": "No content found in specified sections."}
            return {"status": "success", "data": scraped_content}

        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to retrieve the website content: {str(e)}"}
