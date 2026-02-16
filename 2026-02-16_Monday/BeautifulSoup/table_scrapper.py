from bs4 import BeautifulSoup
import requests
import time
import random
import pandas as pd

# Without a Session (requests.get)-Every request is a unique TCP/SSL Handshake
# With a Session (requests.Session)-Implements Connection Pooling and Keep-Alive
def connect_session():
    session = requests.Session()
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"}
    session.headers.update(headers)
    return session

def fetch_page(session, url):
    try:
        response = session.get(url,timeout=10)
        response.raise_for_status()
        if response.status_code == 200:
            print("Fetched page successfully")
        return response.text
    except Exception as e:
        print("Fetch Error : ",e)
        return None

def parse_table(html_data):
    soup = BeautifulSoup(html_data, 'html.parser')
    table = soup.find('table') #search DOM and find <table> tag

    if table is None:
        return []

    page_data = []
    rows= table.find_all('tr')
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    for row in rows:
        cols = row.find_all('td')
        row_data = [col.text.strip() for col in cols]

        if row_data:
            page_data.append(row_data)
    return page_data, headers

def scrape_pagination(session,url,start_page=1,end_page=5):
    all_data = []
    headers = []
    for page in range(start_page,end_page+1):
        print(f"Scraping page {page} : ")
        url = url.format(page)
        html = fetch_page(session,url)

        if html:
            page_data, headers = parse_table(html)
            all_data.extend(page_data)
        time.sleep(random.uniform(1,3))
        print(f'Page {page} scraped Successfully.....')
    return all_data, headers

def save_data(data,headers,filename="table_data.csv"):
    df = pd.DataFrame(data, columns = headers)
    df.to_csv(filename,index=False)
    return df


if __name__ == '__main__':
    url='https://www.scrapethissite.com/pages/forms/?page_num={}'
    session = connect_session()
    # print("Fetching Single page :")
    # html = fetch_page(session, url)
    # if html:
    #     parsed_data = parse_table(html)
    #     for data in parsed_data[:5]:
    #         print(data)
    # else:
    #     print("No data found")

    try:
        data, headers = scrape_pagination(session=session, url=url, start_page=1, end_page=5)
        # print(headers)
        df = save_data(data, headers)
        # print(df)
        # print(data)
        print("Data Saved Successfully ...\n")
    finally:
        session.close()
        print("Session closed.")






