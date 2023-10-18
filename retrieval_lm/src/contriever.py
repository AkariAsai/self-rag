# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import transformers
from transformers import BertModel, XLMRobertaModel

from src import utils


class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb


class XLMRetriever(XLMRobertaModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]
        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb


def load_retriever(model_path, pooling="average", random_init=False):
    # try: check if model exists locally
    path = os.path.join(model_path, "checkpoint.pth")
    if os.path.exists(path):
        pretrained_dict = torch.load(path, map_location="cpu")
        opt = pretrained_dict["opt"]
        if hasattr(opt, "retriever_model_id"):
            retriever_model_id = opt.retriever_model_id
        else:
            # retriever_model_id = "bert-base-uncased"
            retriever_model_id = "bert-base-multilingual-cased"
        tokenizer = utils.load_hf(transformers.AutoTokenizer, retriever_model_id)
        cfg = utils.load_hf(transformers.AutoConfig, retriever_model_id)
        if "xlm" in retriever_model_id:
            model_class = XLMRetriever
        else:
            model_class = Contriever
        retriever = model_class(cfg)
        pretrained_dict = pretrained_dict["model"]

        if any("encoder_q." in key for key in pretrained_dict.keys()):  # test if model is defined with moco class
            pretrained_dict = {k.replace("encoder_q.", ""): v for k, v in pretrained_dict.items() if "encoder_q." in k}
        elif any("encoder." in key for key in pretrained_dict.keys()):  # test if model is defined with inbatch class
            pretrained_dict = {k.replace("encoder.", ""): v for k, v in pretrained_dict.items() if "encoder." in k}
        retriever.load_state_dict(pretrained_dict, strict=False)
    else:
        retriever_model_id = model_path
        if "xlm" in retriever_model_id:
            model_class = XLMRetriever
        else:
            model_class = Contriever
        cfg = utils.load_hf(transformers.AutoConfig, model_path)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_path)
        retriever = utils.load_hf(model_class, model_path)

    return retriever, tokenizer, retriever_model_id
