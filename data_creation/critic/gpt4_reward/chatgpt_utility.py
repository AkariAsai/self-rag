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


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


PROMPT_DICT = {
    "context": (
        "Given an instruction and an output, rate whether the response appears to be a helpful and informative answer to the query, from 1 (lowest) - 5 (highest). We call this score perceived utility.\n\n"
        "The detailed criterion is as follows:\n"
        "5: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs.\n"
        "4: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence. \n"
        "3: The response is acceptable, but some major additions or improvements are needed to satisfy users' needs.\n"
        "2: The response still addresses the main request, but it is not complete or not relevant to the query.\n"
        "1: The response is barely on-topic or completely irrelevant.\n"
        "##\n Instruction: Who is the current prime minister of the UK as of 2023?\n"
        "Output: Boris Johnson was the prime minister of the UK from 2019 - 2022.\n"
        "Perceived utility: 2\n"
        "Explanation: While the output provides a factually correct statement about the UK prime minister from 2019 to 2022, this instruction asks who the prime minister is as of 2023, so it doesn't answer the instruction. Therefore, the utility is 2.\n\n"
        "##\nInstruction: Given a description of a travel destination, recommend 10 tourist attractions with detailed explanations of each. The travel destination is Tokyo, Japan.\n"
        "Output: 'Tokyo is a vibrant city full of exciting tourist attractions. Some must-see sights include the Tokyo Skytree, Tokyo Disneyland, Sensoji Temple, Meiji Shrine, Tsukiji Fish Market, Harajuku, and Shinjuku Gyoen.\n"
        "Perceived utility: 3\n"
        "Explanation: This output doesn't provide descriptions of each attraction and the number of the attractions is also less than 10. While this output partially answers the instructions, it doesn't match the instructions strictly. \n\n"
        "##\nInstruction: {instruction}\n"
        "Output:{output}\n"
    ),
}


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def postprocess(results):
    raw_output = results["choices"][0]["message"]["content"]
    if "\nExplanation:" in raw_output:
        explanation = raw_output.split("\nExplanation:")[1]
        score_string = raw_output.split("\nExplanation:")[0]
        score = None
        for i in range(1, 6):
            if str(i) in score_string:
                score = int(i)
        if score is None:
            return "", explanation
        else:
            return score, explanation
    else:
        return "", ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, nargs='+')
    parser.add_argument('--output_file_name', type=str)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--org_name', type=str)
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()

    with open(args.api_key) as f:
        openai.api_key = f.read()[:-1]
    openai.organization = args.org_name
    examples = []
    for input_file in args.input_files:
        if input_file.endswith(".json"):
            examples += json.load(open(input_file))
        else:
            examples += load_jsonlines(input_file)

    for item in examples:
        if len(item["input"]) > 1:
            item["instruction"] = item["instruction"] + " " + item["input"]

    result_list = []
    if args.n is not None:
        examples = random.sample(examples, k=args.n)

    for idx, example in tqdm(enumerate(examples)):
        try:
            results = completions_with_backoff(
                model=args.model_name,
                messages=[
                    {"role": "user",
                        "content": PROMPT_DICT["context"].format_map(example)},
                ],
                request_timeout=60,
                max_tokens=200,
            )
            score, explanation = postprocess(results)
            result_list.append({"input": example, "score": score, "explanation": score,
                               "raw_output": results["choices"][0]["message"]["content"]})
            if idx % 20 == 0:
                print("Input: {}".format(example["instruction"]))
                print("Output: {}".format(example["output"]))
                print("Score: {0} ({1})".format(score, explanation))

        except (APIError, Timeout, APIConnectionError):
            results = "ERROR: API error outputs"
        if idx % 100 == 0:
            with open(args.output_file_name + "_tmp", "w") as outfile:
                json.dump(result_list, outfile)

    with open(args.output_file_name, "w") as outfile:
        json.dump(result_list, outfile)


if __name__ == "__main__":
    main()
