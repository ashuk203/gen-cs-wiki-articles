import pickle
import pdb

embs_dict_path = "data/word2vec_embs/tok2embs_dict.pickle"

if __name__ == '__main__':
    with open(embs_dict_path, 'rb') as f:
        tok_to_embs = pickle.load(f)
        pdb.set_trace()