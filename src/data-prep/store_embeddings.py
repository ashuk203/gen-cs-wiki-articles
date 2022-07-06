import sys
import gensim

import pickle
import json

from tqdm import tqdm
# import pdb

debug = False

docs_path = "/home/aukey2/cs-wiki-data/cs_outlines_18k/article_jsons.txt"
model_path = '/scratch/kcma2/word2vec/word2vec.model'

embs_out_path = "data/w2v_embs_dict.pickle"

word2vec = gensim.models.Word2Vec.load(model_path).wv


def add_embs(emb1, emb2):
    res_emb = None
    if emb1 is None and emb2 is not None:
        res_emb = emb2
    elif emb2 is None and emb1 is not None:
        res_emb = emb1
    else:
        d = len(emb1)
        res_emb = [emb1[i] + emb2[i] for i in range(d)]

    return res_emb


def emb_divide(emb, c):
    for i in range(len(emb)):
        emb[i] /= c


def get_emb(word):
    emb = None

    try:
        query = word.replace(' ', '_')
        emb = word2vec[query]
    except Exception as e:
        if debug:
            print(e)

        sub_words = word.replace('-', ' ').split(' ')

        try:
            for s_w in sub_words:
                new_emb = add_embs(emb, word2vec[s_w])
                emb = new_emb

            emb_divide(emb, len(sub_words))
        except Exception as e:
            if debug:
                print(e)
                # print("Value of emb", type(emb))

    return emb
            

def save_embs(words, out_file_path):
    word_to_emb = {}
    num_missing = 0

    for i in tqdm(range(len(words))):
        w = words[i]
        w_emb = get_emb(w)

        if w_emb is not None:
            word_to_emb[w] = w_emb
        else:
            num_missing += 1
            if debug:
                print("Unable to generate embedding for ", w)
    
    with open(out_file_path, 'wb') as handle:
        pickle.dump(word_to_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    perc_missing = (num_missing / len(words)) * 100
    print("Unable to generate embeddings for", num_missing, "words (", perc_missing, "%)")




if __name__ == '__main__':
    # Get words that need embeddings generated
    words = []

    with open(docs_path) as f:
        for line in f:
            doc = json.loads(line[:-1])
            words.append(doc["keyword"])


    # Compute and save embeddings
    save_embs(words, embs_out_path)

