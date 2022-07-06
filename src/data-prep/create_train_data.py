import json
from sklearn.model_selection import train_test_split

import re

import pdb

data_root_dir = "/home/aukey2/cs-wiki-data/cs_outlines_18k"
articles_file = f"{data_root_dir}/article_jsons.txt"

data_root_dir = "data/cs_train_first_sent"


# sent_end_pattern = re.compile("[\w\)\"]\.")
sent_end_pattern = re.compile("\.")

def get_first_sent(par):
    match = re.search(sent_end_pattern, par)

    if not match:
        return ""

    first_sent = par[:match.start() + 1]
    return first_sent

# sect_sep = " [SECTSEP] "
sect_sep = ", "

count = 0
empty_count = 0

sources = []
targets = []



with open(articles_file) as f_in:
    while True:
        count += 1
    
        # Get next line from file
        article = f_in.readline()

        # if not article or count >= 1000:
        if not article:
            break

        article = json.loads(article[:-1])

        first_sent = get_first_sent(article["summary"])

        if len(first_sent) == 0:
            # print("\t- ", article["summary"])
            empty_count += 1
            continue

        # cur_source = article["keyword"] + ' [DEF] ' + article["summary"]
        # cur_source = article["keyword"] + ': ' + article["summary"]
        # cur_source = article["keyword"] + ':'
        cur_source = article["keyword"] + ': ' + first_sent
        cur_targ = sect_sep.join(article["sections"])
        
        sources.append(cur_source)
        targets.append(cur_targ)

print(f"Unable to get first sentence for {empty_count} articles.")


train_perc = 0.8
test_perc = 0.1
val_perc = 1 - (train_perc + test_perc)

rest_perc = 1 - train_perc
src_train, src_rest, trg_train, trg_rest = train_test_split(sources, targets, test_size=rest_perc)
src_test, src_val, trg_test, trg_val = train_test_split(src_rest, trg_rest, test_size=(val_perc / rest_perc))


# pdb.set_trace()

def save_split(srcs, trgs, name):
    file_root = f"{data_root_dir}/{name}"

    srcs_f_name = file_root + '.source'
    trgs_f_name = file_root + '.target'

    with open(srcs_f_name, "w") as srcs_f, open(trgs_f_name, "w") as trgs_f:
        for i in range(len(srcs)):
            cur_src = srcs[i]
            cur_trg = trgs[i]

            srcs_f.write(cur_src + '\n')
            trgs_f.write(cur_trg + '\n')


save_split(src_train, trg_train, "train")
save_split(src_test, trg_test, "test")
save_split(src_val, trg_val, "val") 