import jsonlines
import json
import numpy as np
import os
from tqdm import tqdm 
from langdetect import detect
import argparse
import random

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def process_tulu_dataset(fp, single_turn_only=True, max_n=1000, dataset_name=False):
    data = load_jsonlines(fp)
    processed_data = []
    for item in tqdm(data):
        messages = item["messages"]
        if single_turn_only is True and len(messages) > 2:
            continue
        # currently only support single turn
        input = messages[0]["content"].replace("\nOutput:", "")
        if len(input.split(" ")) > 500:
            continue 
        if input[-2:] == "\n\n":
            input = input[:-2]
        if input[-1:] == "\n":
            input = input[:-1]
        output = messages[1]["content"]
        if len(output) == 0:
            continue
        if output[0] == "\n":
            output = output[1:]
        if len(output.split(" ")) > 500:
            continue
        if dataset_name == "sharegpt" or "oasst1": 
            try:
                if detect(input) != "en" or detect(output) != "en":
                    # print("multilingual input")
                    continue
            except:
                print("non text input")
        id = item["id"]
        processed_data.append({"input": "", "instruction": input, "output": output, "id": id, "dataset_name": dataset_name})
    if max_n is not None and max_n < len(processed_data):
        processed_data = random.sample(processed_data, k=max_n)
    return processed_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--input_file', type=str,
                        default=None)
    parser.add_argument('--output_file', type=str,
                        default=None)
    parser.add_argument('--data_prefix', type=str,
                        default=None,)
    parser.add_argument('--single_turn_only', action="store_true",)
    args = parser.parse_args()

    processed_data = process_tulu_dataset(args.input_file, single_turn_only=True, max_n=args.n, dataset_name=args.data_prefix)
    with jsonlines.open(args.output_file, 'w') as writer:
        writer.write_all(processed_data)

if __name__ == "__main__":
    main()
