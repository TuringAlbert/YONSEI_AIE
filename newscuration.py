import selenium
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import pandas as pd
import tqdm
import nltk
import gensim as gensim
# from newspaper import Article
import newspaper

chrome_options = Options()
chrome_options.add_argument("--headless")
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def collect_news():
    driver_path = "/Users/juyoungkim/Development/aie/aie_opensource/chromedriver_mac64/chromedriver.exe"
    driver = selenium.webdriver.Chrome(driver_path)
    driver.get('https://news.google.com/home?hl=en-US&ql=US&ceid=US:en')

    wait = WebDriverWait(driver, 10)
    wait.until(EC.visibility_of_element_located((By.XPATH, '//a[text() = "Top stories"]')))

    article_nodes = driver.find_elements(By.TAG_NAME, "article")
    # print("article_nodes : ", article_nodes)
    df = pd.DataFrame(columns=['title', 'source', 'link'])
    print("df :: ", df)

    for article in tqdm.tqdm(article_nodes):
        title = article.find_element(By.XPATH, ".//h4").text
        link = article.find_element(By.XPATH, ".//div/a").get_attribute('href')
        source = article.find_element(By.XPATH, ".//div/span").text

        df = pd.concat([df, pd.DataFrame({"title": title, "source": source, "link": link}, index=[0])],
                       ignore_index=True)

    driver.quit()
    return df


def collect_custome_news():
    news_type = int(input("please select news type [0 : Business, 1 : Technology, 2 : Sports] : "))
    while True:
        if news_type in [0, 1, 2]:
            break
        else:
            news_type = int(
                input("wrong input!\n please re-select news type [0 : Business, 1 : Technology, 2 : Sports] : "))

    driver_path = "/Users/juyoungkim/Development/aie/aie_opensource/chromedriver_mac64/chromedriver.exe"
    driver = selenium.webdriver.Chrome(driver_path)
    driver.get('https://news.google.com/home?hl=en-US&ql=US&ceid=US:en')

    wait = WebDriverWait(driver, 1)
    wait.until(EC.visibility_of_element_located((By.XPATH, '//a[text() = "Your topics"]')))

    article_nodes = driver.find_elements(By.TAG_NAME, "article")
    df = pd.DataFrame(columns=['title', 'source', 'link'])

    for article in tqdm.tqdm(article_nodes):
        title = article.find_element(By.XPATH, ".//h4").text
        link = article.find_element(By.XPATH, ".//div/a").get_attribute('href')
        source = article.find_element(By.XPATH, ".//div/span").text

        df = pd.concat([df, pd.DataFrame({"title": title, "source": source, "link": link}, index=[0])],
                       ignore_index=True)

    driver.quit()
    return df


def fill_news_contents(df):
    df = df.assign(text=None)

    for index, row in tqdm.tqdm(df.iterrows()):
        try:
            news = newspaper.Article(row['link'], language='en')
            news.download()
            news.parse()

            title = row['title'].strip() if row['title'].strip() else news.title
            contents = news.text.strip() if news.text.strip() else title

            df.at[index, 'title'] = title
            df.at[index, 'text'] = contents

        except Exception as e:
            df.at[index, 'text'] = df.at[index, 'title']
            print(e)
            pass
    return df


def train_news(df):
    nltk.download("punkt")
    nltk.download("stopwords")

    stop_words = nltk.corpus.stopwords.words('english')
    print('stop words : ', stop_words)

    preprocessed_articles = []

    for article in df["text"].tolist():
        words = [
            word.lower()
            for sentence in nltk.sent_tokenize(article)
            for word in nltk.word_tokenize(sentence)
            if word.lower() not in stop_words
        ]

        tokenized_articles = words
        preprocessed_articles.append(tokenized_articles)

    model = gensim.models.Word2Vec(preprocessed_articles, vector_size=100, window=5, min_count=5, workers=4)
    return model, preprocessed_articles


def choose_article(df, model, preprocessed_articles, tag):
    print("model : ", model)
    print("preprocessed : ", preprocessed_articles)
    print("model.wv.index_to_key : ", model.wv.index_to_key)
    if tag not in model.wv.index_to_key:
        print(f"there is no articles about {tag}")
        return

    interest_tag_vector = model.wv[tag]

    article_vectors = []
    for article in preprocessed_articles:
        article_vectors = sum([model.wv[word] for word in article if word in model.wv.index_to_key])
        article_vectors.append(article_vectors)

    similarites = {}
    for i, article_vectors in enumerate(article_vectors):
        similarity = model.wv.cosine_similarities(article_vectors, [interest_tag_vector])[0]
        similarites[i] = similarity

    sorted_articles = sorted(similarites, key=lambda x: similarites[x], reverse=True)
    article_title = df.at[sorted_articles[0], "title"]
    article_link = df.at[sorted_articles[0], "link"]
    print(f"The most similar article for '{tag}' is '{article_title}' at {article_link}")


tag = "Ralph"  # input("키워드를 입력하시오 : ")
df = collect_news()
print("df :\n", df)
# print(df.loc[[1], [1]])
print("df of : ", df[["title"]])

df = fill_news_contents(df)
print("df2 : ", df)
model, preprocessed_articles = train_news(df)

choose_article(df, model, preprocessed_articles, tag)
