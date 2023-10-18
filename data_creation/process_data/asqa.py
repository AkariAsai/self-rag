import json
import argparse
import random
import jsonlines
import datasets
from task_instructions import TASK_INST


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--output_file', type=str,
                        default=None, )
    parser.add_argument('--input_file', type=str,
                        default=None, )
    parser.add_argument('--data_prefix', type=str,
                        default=None,)
    args = parser.parse_args()
    input_data = json.load(open(args.input_file))["train"]

    new_data = []
    for sample_id, item in input_data.items():
        question = item["ambiguous_question"]
        q_id = "{0}_{1}".format(args.data_prefix, sample_id)
        dataset_name = args.data_prefix
        instruction = TASK_INST[args.data_prefix] + "## Input:\n\n" + question
        output = item["annotations"][0]["long_answer"]
        new_data.append({"instruction": instruction, "output": output, "input": "",
                        "topic": "", "id": q_id, "dataset_name": dataset_name})

    if args.n is not None:
        new_data = random.sample(new_data, k=args.n)

    with jsonlines.open(args.output_file, 'w') as writer:
        writer.write_all(new_data)


if __name__ == "__main__":
    main()
