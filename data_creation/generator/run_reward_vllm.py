#!/usr/bin/python
# -*- coding: UTF-8 -*-
import random
import torch
import os
import numpy as np
import copy
from tqdm import tqdm
import json
import argparse
from tqdm import tqdm
import jsonlines
from vllm import LLM, SamplingParams

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

PROMPT_DICT = {
    "ground_instruction": (
        "You will be given an task instruction, evidence, and output. Your objective is to assess the extent to which the output is supported by the information presented in the evidence.\n"
        "Rate the level of support on a scale from 1 ( Ignore / Contradictory), 2 (Little support), 3 (Partially supported), 4 (Mostly supported), 5 (Fully supported)."
    ),
    "ground_input": (
        "##\nTask instruction: {instruction}\n"
        "Evidence: {evidence}\n"
        "Output: {output}"
    ),
    "ground_multi_instruction": (
        "You will receive an instruction, evidence, and output, and optional preceding sentences.  If the preceding sentence is given, the output should be the sentence that follows those preceding sentences. Your task is to evaluate if the output is fully supported by the information provided in the evidence, and provide explanations on your judgement\n"
        "Use the following entailment scale to generate a score:\n"
        "[Fully supported] - All information in output is supported by the evidence, or extractions from the evidence. This is only applicable when the output and part of the evidence are almost identical.\n"
        "[Partially supported] - The output is supported by the evidence to some extent, but there is major information in the output that is not discussed in the evidence. For example, if an instruction asks about two concepts and the evidence only discusses either of them, it should be considered a [Partially supported].\n"
        "[No support / Contradictory] - The output completely ignores evidence, is unrelated to the evidence, or contradicts the evidence. This can also happen if the evidence is irrelevant to the instruction.\n\n"
        "Make sure to not use any external information/knowledge to judge whether the output is true or not. Only check whether the output is supported by the evidence, and not whether the output follows the instructions or not.\n\n"
    ),
    "ground_multi_input": (
        "Task instruction: {instruction}\n"
        "Preceding sentences: {preceding_sentences}\n"
        "Output: {target_output}\n"
        "Evidence: {evidence}"
    ),
    "ground_multi_input_wo_preceding": (
        "Task instruction: {instruction}\n"
        "Output: {target_output}\n"
        "Evidence: {evidence}"
    ),
    "retrieval_instruction": (
        "When provided with instruction, please evaluate whether seeking additional information from external sources such as the web (e.g., Wikipedia) aids in producing a more comprehensive response. Respond with either [Retrieval] or [No Retrieval]."
    ),
    "retrieval_input": (
        "Task instruction: {instruction}"
    ),
    "retrieval_multi_instruction": (
        "You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences.  Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. If the output sentence can be verified solely with the evidence or doesn’t require any verification, respond with [No Retrieval]. If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments.\n\n"
    ),
    "retrieval_multi_input": (
        "Task instruction: {instruction}\n"
        "Preceding sentences: {preceding_sentences}\n"
        "Evidence: {evidence}\n"
        "Output: {target_output}"
    ),
    "multi_retrieval_three_way_instruction": (
        "You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences.  Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. There are three cases:\n"
        "- If the output sentence can be verified solely with the evidence, then respond with [Continue to Use Evidence]. \n"
        "- If the sentence doesn't require any factual verification (e.g., a subjective sentence or a sentence about common sense), then respond with  [No Retrieval]. \n"
        "If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments. \n\n"
    ),
    "multi_retrieval_three_way_input": (
        "Task instruction: {instruction}\n"
        "Preceding sentences: {preceding_sentences}\n"
        "Evidence: {evidence}\n"
        "Output: {target_output}"
    ),
    "multi_retrieval_three_way_input_wo_preceding": (
        "Task instruction: {instruction}\n"
        "Preceding sentences: {preceding_sentences}\n"
        "Evidence: {evidence}\n"
        "Output: {target_output}"
    ),
    "relevance_instruction": (
        "When given instruction and evidence, evaluate whether the evidence is relevant to the instruction and provides valuable information for generating meaningful responses.\n"
        "Use a rating of [Relevant] to indicate relevance and usefulness, and [Irrelevant] to indicate irrelevance."
    ),
    "relevance_input": (
        "Task instruction: {instruction}\n"
        "Evidence: {evidence}"
    ),
    "utility_instruction": (
        "Given an instruction and an output, rate whether the response appears to be a helpful and informative answer to the query, from 1 (lowest) - 5 (highest). We call this score perceived utility.\n"
        "[Utility:5]: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs.\n"
        "[Utility:4]: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence. \n"
        "[Utility:3]: The response is acceptable, but some major additions or improvements are needed to satisfy users' needs.\n"
        "[Utility:2]: The response still addresses the main request, but it is not complete or not relevant to the query.\n"
        "[Utility:1]: The response is barely on-topic or completely irrelevant.\n"
    ),
    "utility_input": (
        "Task instruction: {instruction}\n"
        "Output: {output}"
    ),
}

seed = 633

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print('Cuda:', torch.cuda.is_available())
print('pwd', os.getcwd())


def posprocess_output(answer, return_score=False):
    answer = answer.replace("</s>", "")
    answer = answer.replace("<unk>", "")
    answer = answer.replace("[PAD]", "")
    return answer


def call_model(prompts, model, max_new_tokens=50):
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
    preds = model.generate(prompts, sampling_params)
    preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
    postprocessed_preds = [posprocess_output(pred) for pred in preds]
    return postprocessed_preds, preds


def accuracy(prediction, ground_truth):
    for gt in ground_truth:
        if prediction == gt:
            return 1
        else:
            return 0


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def process_data(input_data, inst_mode, input_mode, split="train", multi_retrieval=False):
    if split == "train":
        prompt = ALPACA_PROMPT_DICT["prompt_input"].format_map(input_data)
        output = str(input_data["output"])
        return prompt, output
    else:
        instruction = PROMPT_DICT[inst_mode]
        if multi_retrieval is True and (input_data["sent_idx"] == 0 or "preceding_sentences" not in input_data or type(input_data["preceding_sentences"]) is not str or len(input_data["preceding_sentences"]) == 0):
            input = PROMPT_DICT[input_mode +
                                "_no_preceding"].format_map(input_data)
        else:
            input = PROMPT_DICT[input_mode].format_map(input_data)
        prompt = ALPACA_PROMPT_DICT["prompt_input"].format_map(
            {"instruction": instruction, "input": input})
        output = "None"
        return prompt, output


def process_data(input_data, inst_mode, input_mode, split="train", multi_retrieval=False):
    if split == "train":
        prompt = ALPACA_PROMPT_DICT["prompt_input"].format_map(input_data)
        # instruction = PROMPT_DICT[inst_mode]
        # input = PROMPT_DICT[input_mode].format_map(input_data)
        # prompt = ALPACA_PROMPT_DICT["prompt_input"].format_map({"instruction": instruction, "input": input})
        output = str(input_data["output"])
        return prompt, output
    else:
        instruction = PROMPT_DICT[inst_mode]
        if multi_retrieval is True and (input_data["sent_idx"] == 0 or "preceding_sentences" not in input_data or type(input_data["preceding_sentences"]) is not str or len(input_data["preceding_sentences"]) == 0):
            input = PROMPT_DICT[input_mode +
                                "_no_preceding"].format_map(input_data)
        else:
            input = PROMPT_DICT[input_mode].format_map(input_data)
        prompt = ALPACA_PROMPT_DICT["prompt_input"].format_map(
            {"instruction": instruction, "input": input})
        output = "None"
        return prompt, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--alias', type=str)
    parser.add_argument('--inst_mode', type=str, default="alpaca")
    parser.add_argument('--input_mode', type=str, default="alpaca")
    parser.add_argument('--n_examples', type=int, default=15)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--sample', type=int, default=0,
                        help="if 0, use all examples")
    parser.add_argument('--top_k', type=int, default=1,
                        help="# of the ctxs to be used")
    parser.add_argument('--int8bit', action="store_true")
    parser.add_argument('--k_shots', type=int, default=0)
    parser.add_argument('--save_prompt', action="store_true")
    parser.add_argument('--use_llama', action="store_true")
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--result_fp', type=str)
    parser.add_argument('--prev_result_fp', type=str, default=None)
    parser.add_argument('--ft_model', action="store_true")
    parser.add_argument('--instruction', type=str)
    parser.add_argument('--metric', type=str, default="f1")
    parser.add_argument('--score_token', type=str, default=None)
    parser.add_argument('--retrieve_mode', type=str, default="vanilla")
    parser.add_argument('--task', type=str, default="relevance")
    parser.add_argument('--parallel', type=str,
                        help="string of format 'i.n_workers' where i is the index of the worker")
    parser.add_argument('--save_score', action="store_true")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for question encoding")
    parser.add_argument('--download_dir', type=str, help="specify download dir",
                        default="/gscratch/h2lab/akari/model_cache")

    args = parser.parse_args()

    gpt = args.model_name
    model = LLM(model=gpt, download_dir=args.download_dir)

    input_path = args.input_file
    if input_path.endswith(".json") or ".json_" in input_path:
        input_data = json.load(open(input_path))
    else:
        input_data = load_jsonlines(input_path)

    if args.split == "train":
        input_data = [item for item in input_data if item["task"] == args.task]

    if args.prev_result_fp is not None:
        prev_results = json.load(open(args.prev_result_fp))
        input_data = input_data[len(prev_results["preds"]):]
    else:
        prev_results = None

    preds = [] if prev_results is None else prev_results["preds"]
    predicted_results = []
    # main loop
    if args.split == "train":
        correct, total = 0, 0
    for idx in tqdm(range(len(input_data) // args.batch_size)):
        batch = input_data[idx*args.batch_size:(idx+1)*args.batch_size]
        processed_batch = [process_data(
            item, inst_mode=args.inst_mode, input_mode=args.input_mode, split=args.split)[0] for item in batch]
        posprocess_output = [process_data(
            item, inst_mode=args.inst_mode, input_mode=args.input_mode, split=args.split)[1] for item in batch]
        preds, raw_edits = call_model(
            processed_batch, model=model, max_new_tokens=args.max_new_tokens)
        for j, item in enumerate(batch):
            if j > len(preds) - 1:
                print("predictions missing")
                print(len(batch))
                print(j)
                continue
            pred = preds[j]
            item["pred"] = pred
            if args.split == "train":
                item["output"] = posprocess_output[j]
                if len(item["pred"]) != "" and item["pred"] == item["output"] or item["pred"] in item["output"] or item["output"] in item["pred"]:
                    item["correct"] = 1.0
                else:
                    print("pred: {0} output: {1}".format(
                        item["pred"], item["output"]))
                    item["correct"] = 0.0
            predicted_results.append(copy.deepcopy(item))
        with open(args.result_fp + "_tmp", "w") as outfile:
            json.dump(predicted_results, outfile)

    if len(input_data) % args.batch_size > 0:
        batch = input_data[(idx+1)*args.batch_size:]
        processed_batch = [process_data(
            item, inst_mode=args.inst_mode, input_mode=args.input_mode, split=args.split)[0] for item in batch]
        posprocess_output = [process_data(
            item, inst_mode=args.inst_mode, input_mode=args.input_mode, split=args.split)[1] for item in batch]
        preds, raw_edits = call_model(
            processed_batch, model=model, max_new_tokens=args.max_new_tokens)
        for j, item in enumerate(batch):
            pred = preds[j]
            item["pred"] = pred
            if args.split == "train":
                item["output"] = posprocess_output[j]
                if len(item["pred"]) != "" and item["pred"] == item["output"] or item["pred"] in item["output"] or item["output"] in item["pred"]:
                    item["correct"] = 1.0
                else:
                    print("pred: {0} output: {1}".format(
                        item["pred"], item["output"]))
                    item["correct"] = 0.0

    if args.split == "train":
        print(np.mean([item["correct"]
              for item in input_data if item["pred"] != ""]))

    with open(args.result_fp, "w") as outfile:
        json.dump(predicted_results, outfile)


if __name__ == "__main__":
    main()
