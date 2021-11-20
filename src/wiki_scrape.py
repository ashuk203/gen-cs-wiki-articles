import threading

import re
import json
import pandas as pd

import wikipedia
import wikipediaapi 

from tqdm import tqdm

TRAIN_OUT_DIR = "data/train/"
KEYWORDS_FILE = "data/Keywords-Springer-83K.csv"

api_driver = wikipediaapi.Wikipedia('en')

class Wikipedia:
    par_delim = re.compile(r"\.[a-zA-Z]")

    def __init__(self, keyword): 
        self.page_title = Wikipedia._get_page_title(keyword)
        self.page = api_driver.page(self.page_title)

    def get_summary(self):
        """
            Returns first paragraph in the wikipedia summary section. 
        """
        full_summary = self.page.summary

        first_p = full_summary.split("\n")[0]
        first_p = re.split(Wikipedia.par_delim, first_p)[0] + "."

        return first_p


    def get_sections(self):
        return [s.title for s in self.page.sections]


    @staticmethod
    def _get_page_title(query):
        """
            Returns the title of the top matching wikipedia page. 
        """
        wiki_page_title = query
        try:
            # Check to see if 
            wikipedia.summary(wiki_page_title, auto_suggest=False)
        except Exception as e:
            # TODO: Confirm error is from page not found

            # Return top matching page 
            wiki_page_title = wikipedia.suggest(query)

        return wiki_page_title


history_lock = threading.Lock()

seen_page_ids = set()
skipped_words = []
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

    res["keyword"] = keyword
    res["index"] = index
    res["summary"] = kw_wiki.get_summary()
    res["sections"] = kw_wiki.get_sections()
    return res 


def worker(thread_id, num_workers, keywords, pbar):
    worker_out_file = f"{TRAIN_OUT_DIR}/{thread_id}.txt"

    total_keywords = 0
    with open(worker_out_file, "w") as f:
        for i in range(thread_id, len(keywords), num_workers):
            cur_keyword = keywords[i]
            train_point = get_train_point(i, cur_keyword)

            if train_point is not None:
                train_point = json.dumps(train_point)
                f.write(train_point + "\n")

            total_keywords += 1
            pbar.update(1)


if __name__ == '__main__':
    keywords_df = pd.read_csv(KEYWORDS_FILE)
    keywords = keywords_df["keyword"]
    keywords = keywords[:30]

    num_threads = 2
    threads = []

    pbar = tqdm(total=len(keywords))
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i, num_threads, keywords, pbar))
        t.start()

        threads.append(t)

    print("Started threads. Waiting for worker threads....")
    for t in threads:
        t.join()

    pbar.close()
    print("Done. Skipped keywords: ", skipped_words)
