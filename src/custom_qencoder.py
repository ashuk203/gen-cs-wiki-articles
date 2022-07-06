import torch
from torch import Tensor, nn
# import torch.nn as nn

from typing import Optional

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRConfig, DPRQuestionEncoderTokenizer
from transf_utils import DPRQuestionEncoderOutput

import pdb
import pickle


model_path = "/scratch/aukey2/rag-ft-models/blank/aug_qencoder"
w2v_embs_dict_path = "/home/aukey2/gen-cs-wiki-articles/rag_branch/data/word2vec_embs/tok2embs_dict.pickle"

class DPRAugQuestionEncoder(DPRQuestionEncoder):
    def __init__(self, config: DPRConfig):
        super().__init__(config)

        self.def_delim_tok = 1024
        self.config = config

        self.question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        with open(w2v_embs_dict_path, 'rb') as f:
            self.tok_to_w2v_embs = pickle.load(f)

        # Initialize weights and apply final processing
        self.post_init()

        self.D_in = 768
        self.D_w2v = 100 
        self.D_out = 868   # Dimension of context doc representations

        self.emb_W = nn.Linear(self.D_in + self.D_w2v, self.D_out)

        # self.const_emb = torch.rand((1, self.D_w2v))


    def get_w2v_embs(self, input_ids):
        """
            Gets word2vec embedding for corresponding input ids
        """

        batch_size = input_ids.shape[0]
        w2v_embs = torch.zeros(batch_size, self.D_w2v)

        for i in range(batch_size):
            key_q = input_ids[i].tolist()
            delim_idx = key_q.index(self.def_delim_tok)

            key_q = tuple(key_q[:delim_idx])

            w2v_emb = self.tok_to_w2v_embs[key_q]["embedding"]
            w2v_embs[i] = torch.tensor(w2v_emb)

        return w2v_embs


    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # print(input_ids)
        # pdb.set_trace()
        outputs = self.question_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        embeddings = outputs.pooler_output
        N = embeddings.shape[0]

        # Augment embeddings with word2vec embeddings
        # temp_w2v = torch.cat([self.const_emb] * N, 0)

        w2v_embs = self.get_w2v_embs(input_ids)
        embeddings = torch.cat([embeddings.cuda(), w2v_embs.cuda()], 1)

        embeddings = self.emb_W(embeddings.cuda())

        return DPRQuestionEncoderOutput(
            pooler_output=embeddings, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions
        )


if __name__ == '__main__':
    question = "transfer learning: transfer learning is blah blah blah"
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    # model = DPRAugQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    config  = DPRConfig.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    model = DPRAugQuestionEncoder(config)

    input_ids = tokenizer(question, return_tensors="pt")["input_ids"]
    outputs = model(input_ids)

    pdb.set_trace()

    # torch.save(model.state_dict(), "models/test/aug_qencoder_test")

    # model.save_pretrained(model_path)
    # torch.save(model, model_path)
    # torch.save({
    #     'epoch': 0,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': None,
    #     'loss': 0,
    # }, model_path)
