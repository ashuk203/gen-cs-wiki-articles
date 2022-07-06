import pickle
import pdb

embs_dict_path = "data/word2vec_embs/w2v_embs_dict.pickle"

if __name__ == '__main__':
    with open(embs_dict_path, 'rb') as f:
        word_to_emb = pickle.load(f)
        pdb.set_trace()