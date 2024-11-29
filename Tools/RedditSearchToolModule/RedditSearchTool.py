import requests
import time
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage

class RedditSearchTool(AtomicFlow):
    """
    A tool to perform searches on Reddit for posts and comments based on various filters.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client_id = self.flow_config.get("CLIENT_ID")  
        self.client_secret = self.flow_config.get("CLIENT_SECRET")  
        self.user_agent = self.flow_config.get("USER_AGENT", "RedditSearchTool")
        self.token_url = "https://www.reddit.com/api/v1/access_token"
        self.search_url = "https://oauth.reddit.com/search"
        
        self.access_token = self.get_access_token()

    def get_access_token(self):
        auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
        headers = {"User-Agent": self.user_agent}
        data = {"grant_type": "client_credentials"}
        response = requests.post(self.token_url, auth=auth, data=data, headers=headers)
        response.raise_for_status()
        return response.json().get("access_token")

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
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": self.user_agent
        }
        params = {
            "q": query,
            "type": options.get("type", "link"),  
            "sort": options.get("sort", "relevance"),
            "t": options.get("time", "all"),
            "subreddit": options.get("subreddit"),
            "limit": options.get("limit", 10)
        }

        try:
            response = requests.get(self.search_url, headers=headers, params=params)
            response.raise_for_status()
            result = response.json()
            formatted_result = self.format_reddit_response(result)
            return {"status": "success", "data": formatted_result}
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}

    def format_reddit_response(self, result):
        posts = result.get("data", {}).get("children", [])
        formatted_posts = []

        for post in posts:
            data = post.get("data", {})
            formatted_post = {
                "title": data.get("title"),
                "author": data.get("author"),
                "subreddit": data.get("subreddit_name_prefixed"),
                "score": data.get("score"),
                "num_comments": data.get("num_comments"),
                "url": data.get("url"),
                "created_utc": data.get("created_utc")
            }
            formatted_posts.append(formatted_post)

        return formatted_posts

