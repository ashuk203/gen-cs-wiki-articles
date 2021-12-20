import pandas as pd 

WIKI_ARTICLES_FILE = "data/WikiPET.csv"

query_pattern = r'computers'
# query_pattern = f".*{query_pattern}.*"

articles_df = pd.read_csv(WIKI_ARTICLES_FILE)
match_articles = articles_df[articles_df["title"].str.contains(query_pattern, case=False) == True]

print(match_articles)

for index, row in match_articles.iterrows():
    print(row["title"].replace("_", " "))
