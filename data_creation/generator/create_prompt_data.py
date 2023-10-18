import jsonlines
import argparse
import json
import os
import spacy

nlp = spacy.load("en_core_web_sm")
separation_str = "\n\n### Response:\n"
TASK_DATA = ["wow", "fever", "arc_easy", "arc_hard", "obqa", "qrecc", "race", "asqa"]

TASK_INST = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
            "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
            "eli5": "Provide a paragraph-length response using simple words to answer the following question.",  
            "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.", 
            "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.", 
            "arc_hard": "Given four answer candidates, A, B, C and D, choose the best answer choice.", 
            "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.", 
            "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."}

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
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--multi_need_retrieval_pred_files', type=str, nargs="+")
    parser.add_argument('--initial_retrieval_file', type=str, nargs="+")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--num_jobs', type=int, default=4)
    parser.add_argument('--top_n', type=int, default=5)
    args = parser.parse_args()

    if args.input_file.endswith(".json"):
        dpr_data = json.load(open(args.input_file))
    else:
        dpr_data = load_jsonlines(args.input_file)

    qid2need_retrieval = {}
    if args.multi_need_retrieval_pred_files is not None:
        for multi_need_retrieval_pred_file in args.multi_need_retrieval_pred_files:
            need_retrieval_data = json.load(open(multi_need_retrieval_pred_file))
            for item in need_retrieval_data:
                if "q_id" in item and "id" not in item:
                    item["id"] = item["q_id"]
                if "id" in item and "q_id" not in item:
                    item["q_id"] = item["id"]
                qid2need_retrieval.setdefault(item["q_id"], {})
                qid2need_retrieval[item["q_id"]][item["sent_idx"]] = item["pred"]

    batch_size = len(dpr_data) // args.num_jobs
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i in range(args.num_jobs):
        processed_data = []
        dpr_data_i = dpr_data[batch_size*i:batch_size*(i+1)]
        for item in dpr_data_i:
            dataset_name = item["dataset_name"]
            if dataset_name in TASK_DATA:
                instruction = TASK_INST[dataset_name] + "## Input:\n\n" + item["instruction"]
            else:
                instruction = item["instruction"]
            preceding_sentences = item["preceding_sentences"]
            target_output = item["target_output"]
            q_id = item["q_id"]
            output = item["output"]
            sent_idx = item["sent_idx"]
            if sent_idx == 0:
                for p_idx, ctx in enumerate(item["ctxs"][:args.top_n]):
                    evidence = ctx["title"] + "\n" + ctx["text"]
                    processed_data.append({"id": q_id, "instruction": instruction, "output": output, "evidence": evidence, "p_idx": p_idx, 
                                        "target_output": target_output, "preceding_sentences": preceding_sentences, "sent_idx": sent_idx})

            else:
                if q_id in qid2need_retrieval and sent_idx in qid2need_retrieval[q_id] and "No Retrieval" in qid2need_retrieval[q_id][sent_idx]:
                    continue
                for p_idx, ctx in enumerate(item["ctxs"][:args.top_n]):
                    evidence = ctx["title"] + "\n" + ctx["text"]
                    processed_data.append({"id": q_id, "instruction": instruction, "output": output, "evidence": evidence, "p_idx": p_idx, 
                                        "target_output": target_output, "preceding_sentences": preceding_sentences, "sent_idx": sent_idx})

        save_file_jsonl(processed_data, os.path.join(
            args.output_dir, "prompt_data_batch_{}.jsonl".format(i)))

if __name__ == "__main__":
    main()