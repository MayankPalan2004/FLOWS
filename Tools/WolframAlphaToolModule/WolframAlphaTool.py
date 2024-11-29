import requests
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage

class WolframAlphaTool(AtomicFlow):
    """
    A tool to query Wolfram|Alpha's Full Results API for computational knowledge.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app_id = self.flow_config.get("APP_ID")  # Access the APP_ID from configuration
        if not self.app_id:
            raise ValueError("APP_ID is required in the configuration.")
        self.base_url = "https://api.wolframalpha.com/v2/query"

    def run(self, input_message: FlowMessage):
        input_data = input_message.data
        query = input_data.get("query")
        options = input_data.get("options", {})

        if not query:
            response = {'error': 'Query cannot be empty'}
        else:
            response = self.query_wolfram_alpha(query, options)

        reply = self.package_output_message(
            input_message=input_message,
            response=response,
        )
        self.send_message(reply)

    def query_wolfram_alpha(self, query, options):
        params = {
            "input": query,
            "appid": self.app_id,
            "output": options.get("output", "JSON"),
            "format": options.get("format", "plaintext"),
            "podstate": options.get("podstate"),
            "includepodid": options.get("includepodid"),
            "exclude": options.get("exclude"),
            "scantimeout": options.get("scantimeout"),
            "podtimeout": options.get("podtimeout"),
            "formattimeout": options.get("formattimeout"),
            "parsetimeout": options.get("parsetimeout"),
            "reinterpret": options.get("reinterpret"),
            "location": options.get("location"),
            "assumption": options.get("assumption"),
            "async": options.get("async"),
            "reinterpret": options.get("reinterpret"),
            "ip": options.get("ip"),
            "latlong": options.get("latlong"),
            "units": options.get("units", "metric"),
            "width": options.get("width"),
            "maxwidth": options.get("maxwidth"),
            "plotwidth": options.get("plotwidth"),
            "mag": options.get("mag"),
            "fontsize": options.get("fontsize"),
            "background": options.get("background"),
            "foreground": options.get("foreground"),
            "timeout": options.get("timeout"),
            "podindex": options.get("podindex"),
            "scanner": options.get("scanner"),
            "async": options.get("async"),
            "reinterpret": options.get("reinterpret"),
            "ip": options.get("ip"),
            "latlong": options.get("latlong"),
            "units": options.get("units", "metric"),
            "width": options.get("width"),
            "maxwidth": options.get("maxwidth"),
            "plotwidth": options.get("plotwidth"),
            "mag": options.get("mag"),
            "fontsize": options.get("fontsize"),
            "background": options.get("background"),
            "foreground": options.get("foreground"),
            "timeout": options.get("timeout"),
            "podindex": options.get("podindex"),
            "scanner": options.get("scanner"),
            "async": options.get("async"),
            "reinterpret": options.get("reinterpret"),
            "ip": options.get("ip"),
            "latlong": options.get("latlong"),
            "units": options.get("units", "metric"),
            "width": options.get("width"),
            "maxwidth": options.get("maxwidth"),
            "plotwidth": options.get("plotwidth"),
            "mag": options.get("mag"),
            "fontsize": options.get("fontsize"),
            "background": options.get("background"),
            "foreground": options.get("foreground"),
            "timeout": options.get("timeout"),
            "podindex": options.get("podindex"),
            "scanner": options.get("scanner"),
            "async": options.get("async"),
            "reinterpret": options.get("reinterpret"),
            "ip": options.get("ip"),
            "latlong": options.get("latlong"),
            "units": options.get("units", "metric"),
            "width": options.get("width"),
            "maxwidth": options.get("maxwidth"),
            "plotwidth": options.get("plotwidth"),
            "mag": options.get("mag"),
            "fontsize": options.get("fontsize"),
            "background": options.get("background"),
            "foreground": options.get("foreground"),
            "timeout": options.get("timeout"),
            "podindex": options.get("podindex"),
            "scanner": options.get("scanner"),
            "async": options.get("async"),
            "reinterpret": options.get("reinterpret"),
            "ip": options.get("ip"),
            "latlong": options.get("latlong"),
            "units": options.get("units", "metric"),
            "width": options.get("width"),
            "maxwidth": options.get("maxwidth"),
            "plotwidth": options.get("plotwidth"),
            "mag": options.get("mag"),
            "fontsize": options.get("fontsize"),
            "background": options.get("background"),
            "foreground": options.get("foreground"),
            "timeout": options.get("timeout"),
            "podindex": options.get("podindex"),
            "scanner": options.get("scanner"),
            "async": options.get("async"),
            "reinterpret": options.get("reinterpret"),
            "ip": options.get("ip"),
            "latlong": options.get("latlong"),
            "units": options.get("units", "metric"),
            "width": options.get("width"),
            "maxwidth": options.get("maxwidth"),
            "plotwidth": options.get("plotwidth"),
            "mag": options.get("mag"),
            "fontsize": options.get("fontsize"),
            "background": options.get("background"),
            "foreground": options.get("foreground"),
            "timeout": options.get("timeout"),
            "podindex": options.get("podindex"),
            "scanner": options.get("scanner"),
            "async": options.get("async"),
            "reinterpret": options.get("reinterpret"),
            "ip": options.get("ip"),
            "latlong": options.get("latlong"),
            "units": options.get("units", "metric"),
            "width": options.get("width"),
            "maxwidth": options.get("maxwidth"),
            "plotwidth": options.get("plotwidth"),
            "mag": options.get("mag"),
            "fontsize": options.get("fontsize"),
            "background": options.get("background"),
            "foreground": options.get("foreground"),
            "timeout": options.get("timeout"),
            "podindex": options.get("podindex"),
            "scanner": options.get("scanner"),
            "async": options.get("async"),
            "reinterpret": options.get("reinterpret"),
            "reinterpret": options.get("reinterpret"),
            "location": options.get("location"),
            "assumption": options.get("assumption"),
            "units": options.get("units", "metric"),
            "width": options.get("width"),
            "maxwidth": options.get("maxwidth"),
            "plotwidth": options.get("plotwidth"),
            "mag": options.get("mag"),
            "fontsize": options.get("fontsize"),
            "background": options.get("background"),
            "foreground": options.get("foreground"),
            "timeout": options.get("timeout"),
            "podindex": options.get("podindex"),
            "scanner": options.get("scanner"),
            "async": options.get("async"),
            "ip": options.get("ip"),
            "latlong": options.get("latlong")
        }

        params = {k: v for k, v in params.items() if v is not None}

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            result = response.json()
            return {"status": "success", "data": result}
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
 
