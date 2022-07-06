import json
from sklearn.model_selection import train_test_split

import pickle
import pdb
import re

embs_dict_path = "data/word2vec_embs/tok2embs_dict.pickle"
articles_file = "/home/aukey2/cs-wiki-data/cs_outlines_18k/article_jsons.txt"

docs_out_file = "data/cs_docs_18k/docs_40_w2v.csv"
train_data_dir = "data/training_data/cs_train_data_w2v"

# sect_sep = " [SECTSEP] "
sect_sep = ", "
sent_end_pattern = re.compile("\.")

with open(embs_dict_path, 'rb') as f:
    emb_words = pickle.load(f)

emb_words = [v["keyword"] for k,v in emb_words.items()]
emb_words = set(emb_words)


def get_first_sent(par):
    match = re.search(sent_end_pattern, par)

    if not match:
        return ""

    first_sent = par[:match.start() + 1]
    return first_sent


def save_split(srcs, trgs, name):
    file_root = f"{train_data_dir}/{name}"

    srcs_f_name = file_root + '.source'
    trgs_f_name = file_root + '.target'

    with open(srcs_f_name, "w") as srcs_f, open(trgs_f_name, "w") as trgs_f:
        for i in range(len(srcs)):
            cur_src = srcs[i]
            cur_trg = trgs[i]

            srcs_f.write(cur_src + '\n')
            trgs_f.write(cur_trg + '\n')
    
    print("Saved", len(srcs), "in", name, "split")


if __name__ == '__main__':

    # Gather all labeled examples
    count = 0
    articles = []

    with open(articles_file) as f_in:
        while True:
            count += 1
        
            article = f_in.readline()

            if not article: # or count >= 1000:
                break

            article = json.loads(article[:-1])

            if article["keyword"] in emb_words:
                articles.append(article)
            # outl = outl_f.readline()[:-1]

    print("Total of", len(articles), "docs available")

    # Docs for knowledge base
    kb_perc = 0.40
    kb_articles, rest_articles = train_test_split(articles, test_size=(1 - kb_perc))

    with open(docs_out_file, "w") as f_out:
        for article in kb_articles:
            outline = sect_sep.join(article["sections"])
            artic_fields = [
                article["keyword"],
                outline,
                article["summary"]
            ]

            res_line = '\t'.join(artic_fields)
            f_out.write(res_line + '\n')

    print("Saved", len(kb_articles), "in knowledgebase")

    # Docs for training (train, test, val split)
    sources = []
    targets = []

    for article in rest_articles:
        first_sent = get_first_sent(article["summary"])
        cur_source = article["keyword"] + ': ' + first_sent

        cur_targ = sect_sep.join(article["sections"])

        sources.append(cur_source)
        targets.append(cur_targ)


    train_perc = 0.8
    test_perc = 0.18
    val_perc = 1 - (train_perc + test_perc)

    rest_perc = 1 - train_perc
    src_train, src_rest, trg_train, trg_rest = train_test_split(sources, targets, test_size=rest_perc)
    src_test, src_val, trg_test, trg_val = train_test_split(src_rest, trg_rest, test_size=(val_perc / rest_perc))


    save_split(src_train, trg_train, "train")
    save_split(src_test, trg_test, "test")
    save_split(src_val, trg_val, "val") 