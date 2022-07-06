from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizerFast
import pickle

from tqdm import tqdm

data_root_dir = "data/word2vec_embs"
embs_dict_path = f"{data_root_dir}/w2v_embs_dict.pickle"
tok_embs_path = f"{data_root_dir}/ctx_tok2embs_dict.pickle"

with open(embs_dict_path, 'rb') as f:
    word_to_emb = pickle.load(f)

ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
tokenizer = ctx_tokenizer
# tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")



def get_toks(query):
    tokens = tokenizer(query, return_tensors="pt")["input_ids"]
    tokens = tokens[0, :-1].tolist()
    tokens = tuple(tokens)

    return tokens


if __name__ == '__main__':
    tok_to_embs = {}

    pbar = tqdm(total = len(word_to_emb))
    for word in word_to_emb:
        w_toks = get_toks(word)
        w_emb = word_to_emb[word]

        tok_to_embs[w_toks] = {
            'keyword': word,
            'embedding': w_emb
        }

        pbar.update(1)
        # if len(tok_to_embs) > 10:
        #     break

    with open(tok_embs_path, 'wb') as handle:
        pickle.dump(tok_to_embs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pbar.close()

