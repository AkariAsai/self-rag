import argparse
import numpy as np
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
from utils import load_file, TASK_INST, PROMPT_DICT, save_file_jsonl, process_arc_instruction, postprocess_answers_closed
from metrics import metric_max_over_ground_truths, exact_match_score, match
import ast
import backoff
import openai
from openai.error import APIError, Timeout, APIConnectionError
ORG_KEY="YOUR_ORG_KEY"

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def completions_instructgpt_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def call_model_chatgpt(prompt, model, max_tokens=50):
    print(model)
    try:
        results = completions_with_backoff(
            model=model,
            messages=[
                {"role": "user",
                    "content": prompt},
            ],
            request_timeout=60,
            max_tokens=max_tokens,
        )
        result = results["choices"][0]["message"]["content"]
    except (APIError, Timeout, APIConnectionError):
        result = "ERROR: API error outputs"
    return result


def call_model_instructgpt(prompt, model, max_tokens=50):
    try:
        results = completions_instructgpt_backoff(model=model, prompt=prompt, temperature=0.0,
                                                  max_tokens=max_tokens, logprobs=5, top_p=1, frequency_penalty=0.0, presence_penalty=0.0)
        result = results["choices"][0]["text"]
    except (APIError, Timeout, APIConnectionError):
        results = "ERROR: API error outputs"
    return result


def call_model(prompts, model, max_new_tokens=50):
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=max_new_tokens)
    preds = model.generate(prompts, sampling_params)
    preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
    postprocessed_preds = [postprocess_output(pred) for pred in preds]
    return postprocessed_preds, preds


def postprocess_output(pred):
    pred = pred.replace("</s>", "")

    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--retrieval_file', type=str, default=None)
    parser.add_argument('--mode', type=str, default="vanilla")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--int8bit', action="store_true")
    parser.add_argument('--metric', type=str)
    parser.add_argument('--top_n', type=int, default=1,
                        help="number of paragraphs to be considered.")
    parser.add_argument('--result_fp', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--prompt_name', type=str, default="prompt_no_input")
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument("--dtype",  type=str, default=None,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--choices",  type=str, default=None,
                        help="space-separated answer candidates")
    parser.add_argument("--instruction",  type=str,
                        default=None, help="task instructions")
    parser.add_argument('--download_dir', type=str, help="specify download dir",
                        default=".cache")
    parser.add_argument('--api_key', type=str, default=None)
    args = parser.parse_args()
    isOpenAI = True if args.model_name in ["text-davinci-003", "gpt-3.5-turbo-0301", "gpt-3.5-turbo"] else False
    if isOpenAI is False:
        if args.dtype is not None:
            model = LLM(model=args.model_name, download_dir=args.download_dir, dtype=args.dtype,
                        tensor_parallel_size=args.world_size,)
        else:
            model = LLM(model=args.model_name, download_dir=args.download_dir,
                        tensor_parallel_size=args.world_size,)

    input_data = load_file(args.input_file)
    if isOpenAI is True and args.api_key is not None:
        with open(args.api_key) as f:
            openai.api_key = f.read()[:-1]
        openai.organization = ORG_KEY

    # For baseline scripts, we simply load pre-retrieved documents from `retrieval_file` option.
    if args.mode == "retrieval":
        if args.retrieval_file is not None:
            retrieval_data = load_file(args.retrieval_file)
            id2retrieval = {}
            for id, item in enumerate(retrieval_data):
                if "id" not in item:
                    id2retrieval[id] = item["ctxs"][:args.top_n]
                else:
                    id2retrieval[item["id"]] = item["ctxs"][:args.top_n]
            for id, item in enumerate(input_data):
                retrieval_result = id2retrieval[id if "id" not in item else item["id"]]
                evidences = ["[{}] ".format(
                    i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_result)]
                item["paragraph"] = "\n".join(evidences)
        else:
            for id, item in enumerate(input_data):
                retrieval_result = item["ctxs"][:args.top_n]
                evidences = ["[{}] ".format(
                    i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_result)]
                item["paragraph"] = "\n".join(evidences)

    for item in input_data:
        if "golds" not in item:
            if "output" in item:
                item["golds"] = item["output"]
            if "answers" in item:
                item["golds"] = item["answers"]
            if "possible_answers" in item:
                item["golds"] = ast.literal_eval(item["possible_answers"])
            if "answerKey" in item:
                item["golds"] = [item["answerKey"]]

        if "instruction" not in item and "question" in item:
            item["instruction"] = item["question"]

        if args.instruction is not None:
            item["instruction"] = args.instruction + \
                "\n\n### Input:\n" + item["instruction"]
        if args.task == "fever" or args.task == "arc_c":
            item["instruction"] = TASK_INST[args.task] + \
                "\n\n### Input:\n" + item["instruction"]

    final_results = []
    for idx in tqdm(range(len(input_data) // args.batch_size)):
        batch = input_data[idx*args.batch_size:(idx+1)*args.batch_size]
        processed_batch = [
            PROMPT_DICT[args.prompt_name].format_map(item) for item in batch]

        if isOpenAI is True:
            preds = []
            for input_instance in processed_batch:
                if args.model_name == "text-davinci-003":
                    pred = call_model_instructgpt(
                        input_instance, model=args.model_name, max_tokens=args.max_new_tokens)
                if args.model_name == "gpt-3.5-turbo-0301" or args.model_name == "gpt-3.5-turbo":
                    pred = call_model_chatgpt(
                        input_instance, model=args.model_name, max_tokens=args.max_new_tokens)
                preds.append(pred)
        else:
            preds, _ = call_model(
                processed_batch, model=model, max_new_tokens=args.max_new_tokens)
        for j, item in enumerate(batch):
            pred = preds[j]
            item["output"] = postprocess_answers_closed(
                pred, args.task, args.choices)
            item["output"] = pred
            final_results.append(item)

    if len(input_data) % args.batch_size > 0:
        batch = input_data[(idx+1)*args.batch_size:]
        processed_batch = [
            PROMPT_DICT[args.prompt_name].format_map(item) for item in batch]
        if isOpenAI is True:
            preds = []
            for input_instance in processed_batch:
                if args.model_name == "text-davinci-003":
                    pred = call_model_instructgpt(
                        input_instance, model=args.model_name, max_tokens=args.max_new_tokens)
                if args.model_name == "gpt-3.5-turbo-0301" or args.model_name == "gpt-3.5-turbo":
                    pred = call_model_chatgpt(
                        input_instance, model=args.model_name, max_tokens=args.max_new_tokens)
                preds.append(pred)
        else:
            preds, _ = call_model(
                processed_batch, model=model, max_new_tokens=args.max_new_tokens)
        for j, item in enumerate(batch):
            pred = preds[j]
            item["output"] = postprocess_answers_closed(
                pred, args.task, args.choices)
            final_results.append(item)

    for item in input_data:
        if args.metric == "em":
            metric_result = metric_max_over_ground_truths(
                exact_match_score, item["output"], item["golds"])
        elif args.metric == "accuracy":
            metric_result = 1.0 if item["golds"][0] in item["output"] else 0.0
        elif args.metric == "match":
            metric_result = match(item["output"], item["golds"])
        else:
            raise NotImplementedError
        item["metric_result"] = metric_result

    print("overall result: {0}".format(
        np.mean([item["metric_result"] for item in input_data])))

    if args.task == "factscore":
        processed_item = []
        for item in input_data:
            processed_item.append(item)
        save_file_jsonl(processed_item, args.result_fp)
    else:
        save_file_jsonl(input_data, args.result_fp)


if __name__ == "__main__":
    main()
