from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import pdb

tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base", 
    # additional_special_tokens=["[SECTSEP]", "[DEF]"]
    additional_special_tokens=["[BET]", "[DEF]"]
)

# model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
# embeddings = model(input_ids).pooler_output

def get_tokens(text, verbose=True):
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

    if verbose:
        print("Input ids for", text, ':')
        print(input_ids)

    return input_ids


if __name__ == '__main__':
    test_toks1 = get_tokens("machine learning [DEF] ML is a branch of soft computing")
    test_toks2 = get_tokens("data [BET] mining [DEF]")
    test_toks3 = get_tokens("data [BET] structures [DEF]")

    # pdb.set_trace()