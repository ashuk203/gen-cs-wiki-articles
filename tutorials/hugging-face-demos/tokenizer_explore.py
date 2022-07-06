from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizerFast
import pdb

ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
tokenizer = ctx_tokenizer
# tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")


def get_tokens(query):
    tokens = tokenizer(query, return_tensors="pt")["input_ids"]

    # pdb.set_trace()
    return tokens


if __name__ == '__main__':
    test_qs = [
        "description logic",    # tensor([[ 101, 6412, 7961,  102]])
        "magnetic tracking",    # tensor([[ 101, 3698, 4083, 1024,  102]])
        "machine learning:",
        "data mining:",
        "machine mining:",
        "data learning:",
        "::"    # tensor([[ 101, 1024, 1024,  102]])
    ]

    for q in test_qs:
        toks = get_tokens(q)
        print(toks)