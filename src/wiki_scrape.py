# TODO: ignore parentheses in keywords in wiki search as it breaks the API

import threading
import sys

import re
import json
import pandas as pd

import wikipedia
import wikipediaapi 

import urllib.request
import urllib.parse
from bs4 import BeautifulSoup

from tqdm import tqdm

TRAIN_OUT_DIR = "data/train_raw/vlarge"
KEYWORDS_FILE = "data/Keywords-Springer-83K.csv"
ARTICLES_FILE = "data/WikiPETScans/WikiPET-verylarge.csv"
TRAIN_SUMMARY_FILE = "logs/scrape_summary_vlarge.log"

ABBRV_PATTERN = re.compile('\([a-zA-z]+\)')
WHITE_SPACE_PATTERN = re.compile('\s+')

IGNORE_SECTIONS = set([
    'See also',
    'References',
    'External links',
    'Further reading',
    'Bibliography',
    'Notes'
])

api_driver = wikipediaapi.Wikipedia('en')

class Wikipedia:
    def __init__(self, keyword):
        # keyword = re.sub(ABBRV_PATTERN, '', keyword)
        # self.page_title = Wikipedia._get_page_title(keyword)
        self.page_title = keyword
        self.page = api_driver.page(self.page_title)
        self.page_id = self.page.pageid

    first_par_req_url = "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&redirects=1&pageids={}"

    @staticmethod
    def _extract_first_par(full_summary):
        first_p = full_summary.split("\n")[0]
        first_p = re.split(Wikipedia.par_delim, first_p)[0] + "."

        return first_p

    def get_full_summary(self):
        """
            Returns first paragraph in the wikipedia summary section. 
        """

        return self.page.summary
        # return Wikipedia._extract_first_par(full_summary)


    def get_first_par(self):
        # encoded_title = urllib.parse.quote_plus(self.page_title)
        req_url = Wikipedia.first_par_req_url.format(self.page_id)
        html = urllib.request.urlopen(req_url)
        
        htmlParse = BeautifulSoup(html, 'html.parser')
        html_ps = htmlParse.find_all("p")

        for para in html_ps:
            para_text = para.get_text()
            para_text = para_text.replace("\\n", "")
            para_text = re.sub(WHITE_SPACE_PATTERN, ' ', para_text)

            if len(para_text) > 5:
                return para_text


    def get_sections(self):
        return [s.title for s in self.page.sections if s.title not in IGNORE_SECTIONS]

    def get_categories(self):
        return [title for title in self.page.categories]

    @staticmethod
    def _get_page_title(query):
        """
            Returns the title of the top matching wikipedia page. 
        """
        wiki_page_title = query
        try:
            # Check to see if article with current name exists
            wikipedia.summary(wiki_page_title, auto_suggest=False)
        except Exception as e:
            # TODO: Confirm error is from page not found

            # Return top matching page 
            wiki_page_title = wikipedia.suggest(query)

        return wiki_page_title


history_lock = threading.Lock()

seen_page_ids = set()
skipped_words = []
error_words = []
sparse_words = []

def get_train_point(index, keyword):
    res = {}
    kw_wiki = Wikipedia(keyword)

    history_lock.acquire()
    if kw_wiki.page_title is None or kw_wiki.page.pageid in seen_page_ids:
        skipped_words.append(keyword)
        history_lock.release()
        return
    
    seen_page_ids.add(kw_wiki.page.pageid)
    history_lock.release()

    res["page_title"] = kw_wiki.page_title
    res["page_id"] = kw_wiki.page.pageid
    res["keyword"] = keyword
    res["index"] = index
    res["summary"] = kw_wiki.get_first_par()
    res["sections"] = kw_wiki.get_sections()

    if res["summary"] is not None and len(res["sections"]) > 0 and len(res["sections"]) < 20:
        return res 
    else:
        history_lock.acquire()
        sparse_words.append(keyword)
        # print(len(res['sections']), res["summary"] is None)
        history_lock.release()



def worker(thread_id, num_workers, keywords, pbar):
    worker_out_file = f"{TRAIN_OUT_DIR}/{thread_id}.txt"

    total_keywords = 0
    with open(worker_out_file, "w") as f:
        for k_i in range(thread_id, len(keywords), num_workers):
            cur_keyword = keywords[k_i]

            try:
                train_point = get_train_point(k_i, cur_keyword)
            except Exception as e:
                print("Except block catch: ", e, file=sys.stderr)
                error_words.append(cur_keyword)
                continue
            
            if train_point is not None:
                train_point = json.dumps(train_point)
                f.write(train_point + "\n")

            total_keywords += 1
            pbar.update(1)


if __name__ == '__main__':
    keywords_df = pd.read_csv(ARTICLES_FILE, error_bad_lines=False)
    keywords = keywords_df["title"]

    # keywords = keywords[:1500]
    # keywords = ['AFC_Ajax']

    # Throws request rate error num_threads more than 8
    num_threads = 8
    threads = []

    pbar = tqdm(total=len(keywords), file=sys.stdout)
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i, num_threads, keywords, pbar))
        t.start()

        threads.append(t)

    print("Started threads. Waiting for worker threads....", file=sys.stderr)
    for t in threads:
        t.join()

    pbar.close()

    with open(TRAIN_SUMMARY_FILE, "w") as f:
        f.write("Done.\n")
        f.write(f"\nErrored on: \n{error_words}\n")
        f.write(f"\nNot enough info for: \n{sparse_words}\n")
        f.write(f"\nDone. Skipped keywords: \n{skipped_words}\n")

