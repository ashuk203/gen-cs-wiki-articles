import re
import json
import pandas as pd

import wikipedia
import wikipediaapi 

from tqdm import tqdm

TRAIN_OUT_FILE = "data/train_data.txt"
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


seen_page_ids = set()
skipped_words = []
def get_train_point(index, keyword):
    res = {}
    kw_wiki = Wikipedia(keyword)
    if kw_wiki.page_title is None or kw_wiki.page.pageid in seen_page_ids:
        skipped_words.append(keyword)
        return
    
    seen_page_ids.add(kw_wiki.page.pageid)

    res["keyword"] = keyword
    res["index"] = index
    res["summary"] = kw_wiki.get_summary()
    res["sections"] = kw_wiki.get_sections()
    return res 


if __name__ == '__main__':
    keywords_df = pd.read_csv(KEYWORDS_FILE)

    with open(TRAIN_OUT_FILE, "w") as f:
        keywords_df = pd.read_csv(KEYWORDS_FILE)
        with tqdm(total=keywords_df.shape[0]) as pbar:
            for index, row in keywords_df.iterrows():
                train_point = get_train_point(index, row["keyword"])

                if train_point is not None:
                    train_point = json.dumps(train_point)
                    f.write(train_point + "\n")

                pbar.update(1)

    print("Skipped keywords: ", skipped_words)
