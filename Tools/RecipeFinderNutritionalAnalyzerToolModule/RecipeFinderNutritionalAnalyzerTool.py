import requests
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage

class RecipeFinderNutritionalAnalyzerTool(AtomicFlow):
    """
    A tool to find recipes and analyze their nutritional information.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.flow_config.get("API_KEY")
        self.base_url = "https://api.spoonacular.com/recipes/complexSearch"
        self.nutrition_url = "https://api.spoonacular.com/recipes/{id}/nutritionWidget.json"

        if not self.api_key:
            raise ValueError("API_KEY is required for accessing Spoonacular API.")

    def run(self, input_message: FlowMessage):
        input_data = input_message.data
        ingredients = input_data.get("ingredients")
        diet = input_data.get("diet")
        max_results = input_data.get("max_results", 5)

        if not ingredients:
            response = {'error': 'Ingredients list cannot be empty'}
        else:
            response = self.find_recipes(ingredients, diet, max_results)

        reply = self.package_output_message(
            input_message=input_message,
            response=response,
        )
        self.send_message(reply)

    def find_recipes(self, ingredients, diet, max_results):
        params = {
            "apiKey": self.api_key,
            "includeIngredients": ingredients,
            "diet": diet,
            "number": max_results,
            "addRecipeInformation": True,
            "fillIngredients": True,
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            recipes = response.json().get("results", [])
            if not recipes:
                return {"error": "No recipes found with the provided ingredients and dietary preferences."}
            
            detailed_recipes = self.get_nutritional_info(recipes)
            return {"status": "success", "data": detailed_recipes}
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}

    def get_nutritional_info(self, recipes):
        detailed_data = []

        for recipe in recipes:
            recipe_id = recipe.get("id")
            nutrition_data = self.retrieve_nutrition(recipe_id)
            if nutrition_data:
                recipe_info = {
                    "title": recipe.get("title"),
                    "source_url": recipe.get("sourceUrl"),
                    "image": recipe.get("image"),
                    "ingredients": [ing["name"] for ing in recipe["extendedIngredients"]],
                    "nutrition": nutrition_data
                }
                detailed_data.append(recipe_info)
        return detailed_data

    def retrieve_nutrition(self, recipe_id):
        url = self.nutrition_url.format(id=recipe_id)
        params = {
            "apiKey": self.api_key
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            nutrition = response.json()
            return {
                "calories": nutrition.get("calories"),
                "carbs": nutrition.get("carbs"),
                "fat": nutrition.get("fat"),
                "protein": nutrition.get("protein")
            }
        except requests.exceptions.RequestException as e:
            return None
