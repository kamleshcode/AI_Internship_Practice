import requests
import re
from bs4 import BeautifulSoup

def parse_beautiful_soup(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup

def to_get_text(soup):
    for para in soup.find_all('h1'):
        print(para.text)

def to_get_links(url,soup):
    for links in soup.find_all('a'):
        link = links.get('href')
        if link[0] == '#':
            print(url+link)
        elif link!='#':
            print(link)

def fetchAndSave(url, path):
    soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    with open(path,"w",encoding='utf-8') as f:
        f.write(soup.prettify())

def to_get_images(soup):
    """Extracts all image source URLs."""
    for img in soup.find_all('img'):
        src = img.get('src')
        if src:
            print(f"Image Found: {src}")

def to_get_tables(soup):
    """Extracts data from HTML tables."""
    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all(['td', 'th'])
            data = [ele.text.strip() for ele in cols]
            print(f"Table Row: {data}")

def search_by_text(soup, keyword):
    """Finds any tag containing a specific word using Regex."""
    # This finds 'h1', 'p', 'span' etc. that contain your keyword
    results = soup.find_all(string=re.compile(keyword, re.IGNORECASE))
    for res in results:
        print(f"Matched Text: {res.strip()} (Parent Tag: {res.parent.name})")


if __name__ == "__main__":
    url1="https://beautiful-soup-4.readthedocs.io/en/latest/"
    data=parse_beautiful_soup(url1)
    # to_get_text(data)
    # to_get_links(url1,data)
    # fetchAndSave(url1,'data/scraped_data.txt')
    # to_get_images(data)
    # to_get_tables(data)
    # search_by_text(data, "python")




