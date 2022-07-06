from transformers import DPRContextEncoderTokenizerFast

ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")


if __name__ == '__main__':
    test_summ = "Hello friends"
    test_title = ""
    
    ctx_tokenizer(
        documents["title"], test_summ, truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]