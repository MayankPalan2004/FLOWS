import requests
import pandas as pd
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage

class StockMarketInsightsTool(AtomicFlow):
    """
    A tool for retrieving and analyzing stock market data.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.flow_config.get("API_KEY")
        self.base_url = "https://www.alphavantage.co/query"

        if not self.api_key:
            raise ValueError("API_KEY is required for accessing Alpha Vantage.")

    def run(self, input_message: FlowMessage):
        input_data = input_message.data
        symbol = input_data.get("symbol")
        data_type = input_data.get("data_type", "TIME_SERIES_DAILY")  
        indicators = input_data.get("indicators", ["SMA"])  

        if not symbol:
            response = {'error': 'Stock symbol cannot be empty'}
        else:
            response = self.perform_stock_analysis(symbol, data_type, indicators)

        reply = self.package_output_message(
            input_message=input_message,
            response=response,
        )
        self.send_message(reply)

    def perform_stock_analysis(self, symbol, data_type, indicators):
        params = {
            "function": data_type,
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "compact",  
            "datatype": "json"
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if "Time Series (Daily)" not in data:
                return {"error": "Failed to retrieve stock data"}

            df = self.format_data(data)
            analysis_results = self.calculate_indicators(df, indicators)
            return analysis_results
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}

    def format_data(self, data):
        time_series_data = data["Time Series (Daily)"]
        df = pd.DataFrame(time_series_data).T
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        return df

    def calculate_indicators(self, df, indicators):
        results = {}

        if "SMA" in indicators:
            sma = df["close"].rolling(window=20).mean().iloc[-1]
            results["SMA"] = round(sma, 2)

        if "RSI" in indicators:
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            results["RSI"] = round(rsi, 2)


        return {"status": "success", "data": results}
