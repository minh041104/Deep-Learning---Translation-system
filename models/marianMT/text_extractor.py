import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def extract_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        for element in soup.find_all(['script', 'style', 'iframe', 'ad', 'ins', 'aside']):
            element.decompose()

        ad_keywords = ['ad', 'ads', 'advertisement', 'banner', 'popup', 'social', 'share','header','footer','video']
        for keyword in ad_keywords:
            for element in soup.find_all(class_=lambda x: x and keyword in x.lower()):
                element.decompose()
            for element in soup.find_all(id=lambda x: x and keyword in x.lower()):
                element.decompose()

        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines()]
        text = '\n'.join(line for line in lines if line)
        if len(text.strip()) < 200:
            raise Exception("Không đủ nội dung, chuyển sang phương pháp Selenium")

        return text

    except Exception as e:
        print(f"crawl bằng requests thất bại: {str(e)}")
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')

            driver = webdriver.Chrome(options=chrome_options)

            try:
                driver.get(url)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                driver.execute_script("""
                    // Loại bỏ các phần tử quảng cáo phổ biến
                    const adSelectors = [
                        'iframe',
                        '[class*="ad"]',
                        '[class*="ads"]',
                        '[class*="banner"]',
                        '[id*="ad"]',
                        '[id*="ads"]',
                        'ins.adsbygoogle',
                        '.advertisement',
                        '.social-share',
                        '.popup',
                        'header',
                        'footer',
                        'video'
                    ];
                    adSelectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach(el => el.remove());
                    });
                """)

                page_source = driver.page_source
                soup = BeautifulSoup(page_source, 'html.parser')

                for element in soup.find_all(['script', 'style']):
                    element.decompose()

                text = soup.get_text(separator='\n')
                lines = [line.strip() for line in text.splitlines()]
                text = '\n'.join(line for line in lines if line)

                return text

            finally:
                driver.quit()

        except Exception as selenium_error:
            return f"Cả hai phương pháp đều thất bại. Lỗi Selenium: {str(selenium_error)}"