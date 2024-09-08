import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


class HTMLTextExtractor:
    def __init__(self):
        self.session = requests.Session()

    def extract_text(self, url):
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            body = soup.find('body')
            if body:
                content = body.get_text(strip=True)
                return content
            else:
                print(f'No body content found for URL: {url}')
                return None
        except requests.exceptions.RequestException as e:
            print(f'Error occurred while extracting text: {e}')
            return None

    def extract_links(self, url):
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            base_url = urlparse(url).netloc  # 現在のURLのドメインを取得
            links = set()
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('http'):
                    link_url = urlparse(href).netloc  # リンクのドメインを取得
                    if link_url != base_url:  # リンクのドメインが現在のURLのドメインと異なる場合のみ追加
                        links.add(href)
            return ', '.join(links)
        except requests.exceptions.RequestException as e:
            print(f'Error occurred while extracting links: {e}')
            return 'extract_links failed'

    def close(self):
        self.session.close()

    def __call__(self, url, extract_type='text'):
        if extract_type == 'text':
            content = self.extract_text(url)
            if content:
                return content
            else:
                print('Failed to retrieve content.')
                return None
        elif extract_type == 'links':
            links = self.extract_links(url)
            if links:
                return links
            else:
                print('Failed to retrieve links.')
                return None
        else:
            print(f'Invalid extract_type: {extract_type}')
            return None

    def __del__(self):
        self.close()