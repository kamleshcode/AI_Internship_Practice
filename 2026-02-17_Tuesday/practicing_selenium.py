from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

# chrome_driver_path = "chromedriver.exe"
#
# options=Options()
# options.add_argument('--headless')
# options.add_argument('--window-size=1200,800')
#
# service = Service(chrome_driver_path)
#
# driver = webdriver.Chrome(service=service,options=options)
# driver.get("https://docs.python.org/3/")
#
# print(driver.title)
# driver.quit()

def create_driver():
    # Simple Driver
    # driver = webdriver.Chrome()
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    return driver

def open_page(driver,urls):
    # driver.gt(url)
    # time.sleep(3)

    driver.get(urls)
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.TAG_NAME,'body')))

def parse_page(driver):
    tables = driver.find_elements(By.CSS_SELECTOR,'table tbody tr')
    rows = tables.find_elements(By.CSS_SELECTOR,'tr')
    print(rows)
    data = []
    for row in rows:
        print("hello")
        cols = row.find_elements(By.TAG_NAME,'td')
        rows_data=[col.text for col in cols]
        if rows_data:
            data.append(rows_data)
    return data


if __name__ == "__main__":
    url_temp = "https://www.scrapethissite.com/pages/forms/"
    drivers = create_driver()
    try:
        open_page(drivers,url_temp)
        data = parse_page(drivers)
        print(data)
    finally:
        drivers.quit()



