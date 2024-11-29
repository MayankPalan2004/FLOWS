import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from aiflows.base_flows import AtomicFlow
from aiflows.messages import FlowMessage

class SeleniumScrapingTool(AtomicFlow):
    """
    A tool that uses Selenium to scrape content from websites based on CSS selectors.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, input_message: FlowMessage):
        input_data = input_message.data
        website_url = input_data.get("website_url")
        css_element = input_data.get("css_element")
        wait_time = input_data.get("wait_time", 3)  

        if not website_url or not css_element:
            response = {'error': 'Website URL and CSS element selector are required'}
        else:
            response = self.scrape_website(website_url, css_element, wait_time)

        reply = self.package_output_message(
            input_message=input_message,
            response=response,
        )
        self.send_message(reply)

    def scrape_website(self, website_url: str, css_element: str, wait_time: int):
        options = Options()
        options.add_argument("--headless")  
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=options)
        
        try:
            driver.get(website_url)
            time.sleep(wait_time) 
            
            elements = driver.find_elements(By.CSS_SELECTOR, css_element)
            content = [element.text for element in elements if element.text.strip() != ""]
            
            if not content:
                return {"error": "No content found for the specified CSS selector."}
            return {"status": "success", "data": content}
        except Exception as e:
            return {"error": f"Failed to retrieve content: {str(e)}"}
        finally:
            driver.quit()
