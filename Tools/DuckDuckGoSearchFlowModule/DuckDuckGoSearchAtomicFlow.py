from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage
from duckduckgo_search import DDGS

class  DuckDuckGoSearchAtomicFlow(AtomicFlow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.search = DDGS()

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

    def format_results(self, results, search_type):
        formatted_results = []
        for result in results:
            if search_type == "text":
                formatted_results.append({
                    "Title": result.get("title"),
                    "URL": result.get("href"),
                    "Snippet": result.get("body"),
                })
            elif search_type == "images":
                formatted_results.append({
                    "Title": result.get("title"),
                    "Image URL": result.get("image"),
                    "Source URL": result.get("thumbnail"),
                })
            elif search_type == "videos":
                formatted_results.append({
                    "Title": result.get("title"),
                    "Video URL": result.get("url"),
                    "Duration": result.get("duration"),
                    "Source": result.get("source"),
                })
            elif search_type == "news":
                formatted_results.append({
                    "Title": result.get("title"),
                    "URL": result.get("url"),
                    "Published Date": result.get("date"),
                    "Source": result.get("source"),
                })
            elif search_type == "maps":
                formatted_results.append({
                    "Name": result.get("name"),
                    "Address": result.get("address"),
                    "Latitude": result.get("latitude"),
                    "Longitude": result.get("longitude"),
                })
            elif search_type == "suggestions":
                formatted_results.append({
                    "Suggestion": result.get("phrase")
                })
            elif search_type == "translate":
                formatted_results.append({
                    "Translation": result
                })
        return formatted_results

    def perform_search(self, query, search_type, options):
        try:
            if search_type == "text":
                results = self.search.text(
                    keywords=query,
                    region=options.get("region", "wt-wt"),
                    safesearch=options.get("safesearch", "moderate"),
                    timelimit=options.get("timelimit"),
                    backend=options.get("backend", "api"),
                    max_results=options.get("max_results", 10),
                )
            elif search_type == "images":
                results = self.search.images(
                    keywords=query,
                    region=options.get("region", "wt-wt"),
                    safesearch=options.get("safesearch", "moderate"),
                    timelimit=options.get("timelimit"),
                    size=options.get("size"),
                    color=options.get("color"),
                    type_image=options.get("type_image"),
                    layout=options.get("layout"),
                    license_image=options.get("license_image"),
                    max_results=options.get("max_results", 10),
                )
            elif search_type == "videos":
                results = self.search.videos(
                    keywords=query,
                    region=options.get("region", "wt-wt"),
                    safesearch=options.get("safesearch", "moderate"),
                    timelimit=options.get("timelimit"),
                    resolution=options.get("resolution"),
                    duration=options.get("duration"),
                    license_videos=options.get("license_videos"),
                    max_results=options.get("max_results", 10),
                )
            elif search_type == "news":
                results = self.search.news(
                    keywords=query,
                    region=options.get("region", "wt-wt"),
                    safesearch=options.get("safesearch", "moderate"),
                    timelimit=options.get("timelimit"),
                    max_results=options.get("max_results", 10),
                )
            elif search_type == "maps":
                results = self.search.maps(
                    keywords=query,
                    place=options.get("place"),
                    street=options.get("street"),
                    city=options.get("city"),
                    county=options.get("county"),
                    state=options.get("state"),
                    country=options.get("country"),
                    postalcode=options.get("postalcode"),
                    latitude=options.get("latitude"),
                    longitude=options.get("longitude"),
                    radius=options.get("radius"),
                    max_results=options.get("max_results", 10),
                )
            elif search_type == "translate":
                to_lang = options.get("to_lang", "en")
                results = self.search.translate(
                    keywords=query,
                    to=to_lang,
                )
            elif search_type == "suggestions":
                results = self.search.suggestions(keywords=query)
            else:
                return {'error': f'Invalid search type: {search_type}'}

            formatted_results = self.format_results(results, search_type)
            return {'results': formatted_results}

        except Exception as e:
            return {'error': str(e)}
