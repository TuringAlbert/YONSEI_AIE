import nltk
import gensim.downloader
from nltk.corpus import wordnet
import pandas as pd
# import openai
import time
from bs4 import BeautifulSoup
from selenium import webdriver
import requests
import os


## BeautifulSoup
class Soup():
    driver = webdriver.Chrome()
    search_keyword = 'hot man'
    search_url = f'https://www.pinterest.com/search/pins/?q={search_keyword}'
    driver.get(search_url)
    time.sleep(5)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    image_tags = soup.find_all('img', {'src':True})

    if not os.path.exists('pinterest_images'):
        os.makedirs('pinterest_images')

    for idx, img in enumerate(image_tags):
        image_url = img['src']
        image_url = image_url.replace("235x", "736x", 1)
        image_name = f"{search_keyword}_{idx}.jpg"
        image_path = os.path.join('pinterest_images', image_name)

    response = requests.get(image_url)
    with open(image_path, 'wb') as f:
        f.write(response.content)

    driver.quit()

Soup()