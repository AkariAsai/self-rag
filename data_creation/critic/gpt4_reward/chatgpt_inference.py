import openai
import pandas as pd
import argparse
import json
from collections import Counter
from tqdm import tqdm
import backoff
from openai.error import APIError, Timeout, APIConnectionError
import jsonlines
import random
from sacrebleu.metrics import BLEU, CHRF, TER
bleu = BLEU()


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def postprocess(results):
    raw_output = results["choices"][0]["message"]["content"]
    return raw_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file_name', type=str)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--org_name', type=str)
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()

    with open(args.api_key) as f:
        openai.api_key = f.read()[:-1]
    openai.organization = args.org_name
    examples = load_jsonlines(args.input_file)
    result_list = []
    if args.n is not None:
        examples = random.sample(examples, args.n)
    for idx, example in tqdm(enumerate(examples)):
        try:
            results = completions_with_backoff(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user",
                        "content": example["instruction"]},
                ],
                request_timeout=60,
                max_tokens=200,
            )
            pred = postprocess(results)
            metric_result = bleu.corpus_score(
                [pred], [[example["answers"]]]).score
            result_list.append({"input": example, "pred": pred, "bleu": metric_result,
                               "raw_output": results["choices"][0]["message"]["content"]})
            if idx % 20 == 0:
                print("Input: {}".format(example["instruction"]))
                print("gold: {}".format(example["answers"]))
                print("pred: {}".format(pred))
                print(metric_result)

        except (APIError, Timeout, APIConnectionError):
            results = "ERROR: API error outputs"
        if idx % 100 == 0:
            with open(args.output_file_name + "_tmp", "w") as outfile:
                json.dump(result_list, outfile)

    with open(args.output_file_name, "w") as outfile:
        json.dump(result_list, outfile)


if __name__ == "__main__":
    main()
