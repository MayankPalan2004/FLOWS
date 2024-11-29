import requests
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage

class WeatherActivitySuggestionTool(AtomicFlow):
    """
    A tool to suggest activities based on current or forecasted weather.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.flow_config.get("API_KEY")
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

        if not self.api_key:
            raise ValueError("API_KEY is required for accessing the OpenWeatherMap API.")

    def run(self, input_message: FlowMessage):
        input_data = input_message.data
        location = input_data.get("location")
        preferences = input_data.get("preferences", "outdoor") 

        if not location:
            response = {'error': 'Location cannot be empty'}
        else:
            response = self.suggest_activity(location, preferences)

        reply = self.package_output_message(
            input_message=input_message,
            response=response,
        )
        self.send_message(reply)

    def suggest_activity(self, location, preferences):
        params = {
            "q": location,
            "APPID": self.api_key,  
            "units": "metric"
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            weather_data = response.json()

            if "main" not in weather_data or "weather" not in weather_data:
                return {"error": "Incomplete weather data received. Please try again."}

            activity_suggestions = self.generate_activity_suggestions(weather_data, preferences)
            return {"status": "success", "data": activity_suggestions}
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
        except KeyError as e:
            return {"error": f"Missing expected data in response: {str(e)}"}

    def generate_activity_suggestions(self, weather_data, preferences):
        temp = weather_data["main"]["temp"]
        weather_desc = weather_data["weather"][0]["description"]

        if preferences == "outdoor":
            if "rain" in weather_desc or "snow" in weather_desc:
                activities = ["Visit a museum", "Go to a cafe", "Watch a movie"]
            elif temp > 20:
                activities = ["Go for a hike", "Play beach volleyball", "Have a picnic"]
            else:
                activities = ["Jogging", "Outdoor photography", "Visit a park"]
        else:
            activities = ["Yoga", "Reading", "Indoor games like chess or cards"]
        
        return {
            "location": weather_data["name"],
            "temperature": temp,
            "weather_description": weather_desc,
            "suggested_activities": activities
        }
