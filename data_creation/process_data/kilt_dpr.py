import json
import argparse
import random
import jsonlines
from task_instructions import TASK_INST

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--output_file', type=str,
                        default=None, )
    parser.add_argument('--data_prefix', type=str,
                        default=None, )
    args = parser.parse_args()
    input_data = json.load(open(args.input_file))
    
    new_data = []
    for idx, item in enumerate(input_data):
        q_id = "{0}_{1}".format(args.data_prefix, idx)
        instruction = item["question"]
        if args.data_prefix in TASK_INST:
            instruction = TASK_INST[args.data_prefix] + "## Input:\n\n" + instruction
        output = item["answers"][0]
        topic = item["positive_ctxs"][0]["title"]
        if args.data_prefix == "fever":
            if output not in ["REFUTES", "SUPPORTS"]:
                print(output)
            output = "false" if output == "REFUTES" else "true"
        new_data.append({"instruction": instruction, "output": output, "input": "", "topic": topic, "id": q_id, "dataset_name": args.data_prefix})

    if args.n is not None:
        new_data = random.sample(new_data, k=args.n)
    
    with jsonlines.open(args.output_file, 'w') as writer:
        writer.write_all(new_data)

if __name__ == "__main__":
    main()
