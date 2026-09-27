"""
Nike Product Scraper
Crawls Nike product detail pages to extract text and images
"""
import json
import os
import time
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NikeScraper:
    def __init__(self, output_dir="data"):
        self.output_dir = Path(output_dir)
        self.products_dir = self.output_dir / "products"
        self.images_dir = self.output_dir / "images"
        
        self.products_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def get_product_urls(self, max_products=500):
        """Discover Nike product URLs from multiple categories"""
        logger.info("Discovering product URLs...")
        
        categories = [
            "https://www.nike.com/w/mens-shoes-nik1zy7ok",
            "https://www.nike.com/w/womens-shoes-5e1x6zy7ok",
            "https://www.nike.com/w/mens-clothing-6ymx6znik1",
            "https://www.nike.com/w/womens-clothing-5e1x6z6ymx6",
        ]
        
        product_urls = set()
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            )
            page = context.new_page()
            
            for category_url in categories:
                if len(product_urls) >= max_products:
                    break
                    
                logger.info(f"Crawling category: {category_url}")
                
                try:
                    page.goto(category_url, wait_until="networkidle", timeout=30000)
                    
                    # Scroll to load more products
                    for _ in range(10):
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        time.sleep(1.5)
                        
                        if len(product_urls) >= max_products:
                            break
                    
                    # Extract product links
                    content = page.content()
                    soup = BeautifulSoup(content, 'lxml')
                    
                    # Nike product cards typically have links with specific patterns
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        if '/t/' in href or '/product/' in href:
                            full_url = urljoin("https://www.nike.com", href)
                            product_urls.add(full_url)
                            
                            if len(product_urls) >= max_products:
                                break
                    
                    logger.info(f"Found {len(product_urls)} product URLs so far")
                    
                except Exception as e:
                    logger.error(f"Error crawling {category_url}: {e}")
                    continue
            
            browser.close()
        
        logger.info(f"Discovered {len(product_urls)} total product URLs")
        return list(product_urls)[:max_products]
    
    def scrape_product(self, url):
        """Scrape a single product page"""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until="networkidle", timeout=30000)
                
                # Wait for product content
                time.sleep(2)
                
                content = page.content()
                browser.close()
            
            soup = BeautifulSoup(content, 'lxml')
            
            product_data = {
                'url': url,
                'id': self._generate_product_id(url),
                'title': None,
                'subtitle': None,
                'price': None,
                'description': None,
                'category': None,
                'color': None,
                'style_code': None,
                'images': [],
                'features': []
            }
            
            # Extract title
            title_elem = soup.find('h1', {'id': 'pdp_product_title'})
            if title_elem:
                product_data['title'] = title_elem.get_text(strip=True)
            
            # Extract subtitle
            subtitle_elem = soup.find('h2', {'id': 'pdp_product_subtitle'})
            if subtitle_elem:
                product_data['subtitle'] = subtitle_elem.get_text(strip=True)
            
            # Extract price
            price_elem = soup.find('div', {'data-test': 'product-price'})
            if price_elem:
                product_data['price'] = price_elem.get_text(strip=True)
            
            # Extract description
            desc_elem = soup.find('div', {'class': 'description-preview'})
            if desc_elem:
                product_data['description'] = desc_elem.get_text(strip=True)
            
            # Extract color
            color_elem = soup.find('li', {'data-test': 'product-color'})
            if color_elem:
                product_data['color'] = color_elem.get_text(strip=True).replace('Colour Shown:', '').strip()
            
            # Extract style code
            style_elem = soup.find('li', {'data-test': 'product-style'})
            if style_elem:
                product_data['style_code'] = style_elem.get_text(strip=True).replace('Style:', '').strip()
            
            # Extract features
            features = soup.find_all('li', {'class': 'description-preview__list-item'})
            product_data['features'] = [f.get_text(strip=True) for f in features]
            
            # Extract images
            image_elements = soup.find_all('img')
            seen_urls = set()
            
            for img in image_elements:
                img_url = img.get('src') or img.get('data-src')
                if img_url and 'nike.com' in img_url and img_url not in seen_urls:
                    # Filter for high-quality product images
                    if any(x in img_url for x in ['/image/upload/', 'static.nike.com/a/images']):
                        seen_urls.add(img_url)
                        product_data['images'].append(img_url)
            
            return product_data
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def download_image(self, url, product_id, image_index):
        """Download a single product image"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Determine file extension
            content_type = response.headers.get('content-type', '')
            ext = 'jpg'
            if 'png' in content_type:
                ext = 'png'
            elif 'webp' in content_type:
                ext = 'webp'
            
            filename = f"{product_id}_{image_index}.{ext}"
            filepath = self.images_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return str(filepath.relative_to(self.output_dir))
            
        except Exception as e:
            logger.error(f"Error downloading image {url}: {e}")
            return None
    
    def scrape_products(self, product_urls, max_products=500):
        """Scrape multiple product pages"""
        products = []
        
        for i, url in enumerate(tqdm(product_urls[:max_products], desc="Scraping products")):
            product_data = self.scrape_product(url)
            
            if product_data:
                # Download images
                downloaded_images = []
                for idx, img_url in enumerate(product_data['images'][:8]):  # Max 8 images per product
                    local_path = self.download_image(img_url, product_data['id'], idx)
                    if local_path:
                        downloaded_images.append({
                            'url': img_url,
                            'local_path': local_path
                        })
                
                product_data['downloaded_images'] = downloaded_images
                products.append(product_data)
                
                # Save individual product JSON
                product_file = self.products_dir / f"{product_data['id']}.json"
                with open(product_file, 'w') as f:
                    json.dump(product_data, f, indent=2)
            
            # Rate limiting
            time.sleep(1.5)
            
            # Periodic save
            if (i + 1) % 50 == 0:
                self._save_catalog(products)
        
        self._save_catalog(products)
        return products
    
    def _generate_product_id(self, url):
        """Generate a unique product ID from URL"""
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def _save_catalog(self, products):
        """Save the full catalog"""
        catalog_file = self.output_dir / "catalog.json"
        with open(catalog_file, 'w') as f:
            json.dump(products, f, indent=2)
        logger.info(f"Saved catalog with {len(products)} products to {catalog_file}")


def main():
    scraper = NikeScraper(output_dir="data")
    
    # Discover product URLs
    product_urls = scraper.get_product_urls(max_products=500)
    
    # Save URLs
    urls_file = scraper.output_dir / "product_urls.json"
    with open(urls_file, 'w') as f:
        json.dump(product_urls, f, indent=2)
    
    # Scrape products
    products = scraper.scrape_products(product_urls, max_products=500)
    
    print(f"\n✓ Scraped {len(products)} products")
    print(f"✓ Data saved to {scraper.output_dir}")


if __name__ == "__main__":
    main()
