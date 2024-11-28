import requests
import xml.etree.ElementTree as ET
import re
import logging
from datetime import datetime
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage

class ArxivSearchAtomicFlow(AtomicFlow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_url = "http://export.arxiv.org/api/query"
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s][%(name)s:%(lineno)d][%(levelname)s] - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def run(self, input_message: FlowMessage):
        input_data = input_message.data
        response = {}

        query = input_data.get("query", "").strip()
        if not query:
            response["error"] = "Search query cannot be empty."
            self._send_reply(input_message, response)
            return

        max_results = input_data.get("max_results", 5)
        sort_by = input_data.get("sort_by", "relevance")
        sort_order = input_data.get("sort_order", "descending")
        categories = input_data.get("categories", [])
        start_date = input_data.get("start_date")
        end_date = input_data.get("end_date")

        try:
            search_output = self.search_arxiv(query, max_results, sort_by, sort_order, categories, start_date, end_date)
            response["result"] = search_output
        except Exception as e:
            response["error"] = f"Error during arXiv search: {e}"

        self._send_reply(input_message, response)

    def search_arxiv(self, query, max_results, sort_by, sort_order, categories, start_date, end_date):
        search_query = f'all:{query}'
        if categories:
            categories_query = ' OR '.join([f'cat:{cat}' for cat in categories])
            search_query = f'({search_query}) AND ({categories_query})'

        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': sort_by,
            'sortOrder': sort_order
        }

        headers = {'User-Agent': 'ArxivAtomicFlow/1.0 (contact@example.com)'}
        response = requests.get(self.api_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        articles = self._parse_response(response.text, start_date, end_date)
        total_results = len(articles)
        return {"query": query, "total_results": total_results, "articles": articles}

    def _parse_response(self, xml_data, start_date, end_date):
        root = ET.fromstring(xml_data)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        articles = []

        start_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        for entry in root.findall('atom:entry', namespace):
            title = entry.find('atom:title', namespace).text.strip()
            authors = [author.find('atom:name', namespace).text.strip() for author in entry.findall('atom:author', namespace)]
            summary = entry.find('atom:summary', namespace).text.strip()
            published_str = entry.find('atom:published', namespace).text.strip()
            published_date = datetime.strptime(published_str, "%Y-%m-%dT%H:%M:%SZ")

            if start_date and published_date < start_date:
                continue
            if end_date and published_date > end_date:
                continue

            links = entry.findall('atom:link', namespace)
            url = ""
            pdf_url = ""
            for link in links:
                if link.attrib.get('title') == 'pdf':
                    pdf_url = link.attrib.get('href')
                elif link.attrib.get('rel') == 'alternate':
                    url = link.attrib.get('href')

            articles.append({
                "title": title,
                "authors": authors,
                "summary": summary,
                "published": published_str,
                "url": url,
                "pdf_url": pdf_url
            })

        return articles

    def _send_reply(self, input_message, response):
        reply = self.package_output_message(input_message=input_message, response=response)
        self.send_message(reply)
