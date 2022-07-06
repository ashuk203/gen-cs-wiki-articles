from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import pdb

tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

input_ids = tokenizer("Hello, is my dog cute?", return_tensors="pt")["input_ids"]

outputs = model(input_ids, return_dict=False)
pdb.set_trace()

embeddings = outputs.pooler_output

# pdb.set_trace()
print(embeddings)