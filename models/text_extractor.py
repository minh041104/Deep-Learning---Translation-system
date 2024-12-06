import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.88 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        for footer in soup.find_all('footer'):
            footer.decompose()

        content = []
        for tag in ['h1', 'h2', 'h3', 'p','li','ul', 'ol', 'li','i']:
            for element in soup.find_all(tag):
                content.append(element.get_text(strip=True))

        return '\n'.join(content)
    
    except requests.exceptions.RequestException as e:
        return f"Lỗi khi truy cập URL: {e}"

