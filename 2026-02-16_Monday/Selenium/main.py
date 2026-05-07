import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/144.0.0.0")
    return webdriver.Chrome(options=chrome_options)


def save_to_csv(data, filename="my_data.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"File saved as {filename}")

def scrape_with_selenium(driver, base_url, total_pages=3):
    all_data = []
    headers = []
    try:
        for page in range(1, total_pages + 1):
            print(f"Scraping page {page}...")
            driver.get(base_url.format(page))
            wait = WebDriverWait(driver, 10)
            table = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "table")))
            if not headers:
                header_elements = table.find_elements(By.TAG_NAME, "th")
                headers = [th.text.strip() for th in header_elements]
                if headers:
                    all_data.append(headers)
            for row in table.find_elements(By.TAG_NAME, "tr"):
                cols = row.find_elements(By.TAG_NAME, "td")
                if cols:
                    all_data.append([col.text.strip() for col in cols])
            time.sleep(random.uniform(2, 4))
    except Exception as e:
        print(f"Error during scraping: {e}")
    return all_data


if __name__ == "__main__":
    my_driver = get_driver()
    url_template = 'https://www.scrapethissite.com/pages/forms/?page_num={}'
    scraped_data = scrape_with_selenium(my_driver, url_template, total_pages=3)
    if scraped_data:
        save_to_csv(scraped_data)
        print(f"First 5 rows: {scraped_data[:5]}")

    my_driver.quit()
