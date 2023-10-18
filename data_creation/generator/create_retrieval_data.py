from tqdm import tqdm
import jsonlines
import argparse
import json
import os
import spacy
nlp = spacy.load("en_core_web_sm")
TASK_DATA = ["nq", "wow", "fever", "tqa", "arc_easy", "arc_hard", "obqa", "qrecc", "race", "asqa"]
def split_sentences(paragraph):
    doc = nlp(paragraph)
    sentences = []
    for sent in doc.sents:
        sentences.append(sent.text)
    return sentences


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, nargs="+")
    parser.add_argument('--need_retrieval_files', type=str, default=None, nargs="+")
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--initial_retrieval_file', type=str, default=None)
    parser.add_argument('--multiple_sent', action="store_true")
    args = parser.parse_args()
    processed_data = []
    splitted_data = []
    if args.need_retrieval_files is not None:
        retrieval_necessity = {}
        for f in args.need_retrieval_files:
            retrieval_necessity_f = json.load(open(f))
            retrieval_necessity.update({item["id"]: item["pred"] for item in retrieval_necessity_f})
    else:
        retrieval_necessity = None

    id2evidence = None
    if args.initial_retrieval_file is not None:
        if args.initial_retrieval_file.endswith(".json"):
            data = json.load(open(args.initial_retrieval_file))
        else:
            data = load_jsonlines(args.initial_retrieval_file)
        if "id" in data[0]:
            id2evidence = {item["id"]: item["ctxs"][0] for item in data if ("sent_idx" not in item or item["sent_idx"] == 0)}
        elif "q_id" in data[0]:
            id2evidence = {item["q_id"]: item["ctxs"][0] for item in data if ("sent_idx" not in item or item["sent_idx"] == 0)}
            
    for input_file in args.input_files:
        if input_file.endswith(".json"):
            data = json.load(open(input_file))
        else:
            data = load_jsonlines(input_file)

        if args.multiple_sent is True:
            for idx, item in tqdm(enumerate(data)):
                q_id = item["id"]
                instruction = item["instruction"]
                dataset_name = item["dataset_name"]
                if dataset_name in TASK_DATA and "## Input:\n\n" in instruction:
                    # For task data, we remove the task-specific instruction
                    instruction = instruction.split("## Input:\n\n")[1]
                if len(item["input"]) > 0 and item["input"] not in instruction:
                    instruction = instruction + " " + item["input"]
                output = item["output"]
                if retrieval_necessity is not None and q_id in retrieval_necessity and retrieval_necessity[q_id] is False:
                    continue

                splitted_output = split_sentences(output)
                skipped = {}
                for sent_idx in range(len(splitted_output)):
                    if len(splitted_output) > 2 and len(splitted_output[sent_idx]) < 30:
                        skipped[sent_idx] = True
                        continue
                    else:
                        skipped[sent_idx] = False

                    question = instruction + " " + splitted_output[sent_idx]
                    output = splitted_output[sent_idx]
                    if sent_idx > 0:
                        preceding_sentences = " ".join(
                            splitted_output[:sent_idx])
                    else:
                        preceding_sentences = None
                    if "evidence" in item:
                        evidence = item["evidence"]
                    else:
                        if id2evidence is not None and q_id in id2evidence:
                            evidence = id2evidence[q_id]["title"] + "\n" + id2evidence[q_id]["text"]
                        else:
                            evidence = None
                    processed_entry = {"question": question, "answers": [output], "output": item["output"], "target_output": output,
                                       "instruction": instruction, "preceding_sentences": preceding_sentences,
                                       "q_id": q_id, "sent_idx": sent_idx, "evidence": evidence, "dataset_name": dataset_name}
                    splitted_data.append({"q_id": q_id, "dataset_name": dataset_name,
                                     "instruction": instruction, "output": output, "splitted_output": splitted_output, "skipped": skipped})
                    processed_data.append(processed_entry)

        else:
            for idx, item in tqdm(enumerate(data)):
                q_id = item["id"]
                instruction = item["instruction"]
                if item["dataset_name"] in ["nq", "fever", "wow"] and "\n\nInput: " in instruction:
                    instruction = instruction.split("\n\nInput: ")[1]
                output = item["output"]
                # add output for retrieval
                processed_entry = {"question": instruction +
                                   " " + output, "answers": [output], "q_id": q_id, }
                processed_data.append(processed_entry)
    print(len(processed_data))
    save_file_jsonl(processed_data, args.output_file)
    save_file_jsonl(splitted_data, args.output_file + "_splitted")

if __name__ == "__main__":
    main()
