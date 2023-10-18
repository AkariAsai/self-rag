import json
from collections import Counter
import random
import jsonlines
import argparse
from pathlib import Path
import spacy
import os
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
retrieval_tokens_names = ["[No Retrieval]", "[Retrieval]"]
ground_tokens_names = ["[Utility:1]", "[Utility:2]",
                       "[Utility:3]", "[Utility:4]", "[Utility:5]"]
utility_tokens_names = ["[Fully supported]",
                        "[Partially supported]", "[No support / Contradictory]"]
other_special_tokens = ["<s>", "</s>", "[PAD]",
                        "<unk>", "<paragraph>", "</paragraph>"]

def postprocess(pred):
    special_tokens = rel_tokens_names + retrieval_tokens_names + \
        ground_tokens_names + utility_tokens_names + other_special_tokens
    for item in special_tokens:
        pred = pred.replace(item, "")
    pred = pred.replace("</s>", "")
    pred = pred.replace("<unk>", "")
    if len(pred) == 0:
        return ""
    if pred[0] == " ":
        pred = pred[1:]
    pred = pred.replace("  ", " ")
    if len(pred) == 0:
        return ""
    if pred[-1] == " ":
        pred = pred[:-1]
    return pred


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

nlp = spacy.load("en_core_web_sm")


def load_json(fn):
    return json.load(open(fn))


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def split_sentences(paragraph):
    doc = nlp(paragraph)
    sentences = []
    for sent in doc.sents:
        sentences.append(sent.text)
    return sentences


def convert_score_to_utility_token(pred):
    if len(pred) == 0:
        print("Utility empty")
        return None
    
    if "1" in pred or "2" in pred or "3" in pred or "4" in pred or "5" in pred:
        for i in ["1", "2", "3", "4", "5"]:
            if i in pred:
                return "[Utility:{}]".format(i)
        return "[Utility:{}]".format(5)
    if pred in ["1", "2", "3", "4", "5", 1, 2, 3, 4, 5]:
        return "[Utility:{}]".format(pred)
    else:
        if pred[0] != "[":
            pred = "[" + pred
        if pred in ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]:
            return pred
        else:
            print(pred)
            return None


def convert_score_to_retrieval_token(pred):
    if len(pred) == 0:
        print("Retrieve token empty")
        return None
    if pred[0] != "[":
        pred = "[" + pred
    if pred in ["[Retrieval]", "[No Retrieval]", "[Continue to Use Evidence]"]:
        return pred
    elif pred == "Yes" or pred == "[Yes]":
        return "[Retrieval]"
    elif pred == "No" or pred == "[No]":
        return "[No Retrieval]"
    else:
        print(pred)
        # print("not implemented")
        return "[No Retrieval]"

def convert_score_to_groudness(pred):
    if len(pred) == 0:
        return None
    if pred[0] != "[":
        pred = "[" + pred
    if pred in ["[No support / Contradictory]", "[Partially supported]", "[Fully supported]"]:
        return pred
    elif pred in ["4", "5"]:
        return "[Fully supported]"
    else:
        print("invalid groundness")
        print(pred)
        return None


def combine_results(input_data, results, type):
    for item, pred in zip(input_data, results["preds"]):
        item[type] = pred
    return input_data


def postprocess_relevance_reward_token(pred):
    if len(pred) == 0:
        print("relevance token empty")
        return None
    if "Relevant" in pred:
        return "[Relevant]"
    elif "Irrelevant" in pred:
        return "[Irrelevant]"
    else:
        return None

def load_file(file_name):
    print(file_name)
    if file_name.endswith(".json"):
        data = json.load(open(file_name))
    elif file_name.endswith(".jsonl"):
        data = load_jsonlines(file_name)
    else:
        if ".json_" in file_name:
            data = json.load(open(file_name))
        elif ".jsonl_" in file_name:
            data = load_jsonlines(file_name)
    return data

def load_all_files(file_paths):
    final_results = {}
    for fp in file_paths:
        data = load_file(fp)
        for item in data:
            q_id = item["id"] if "id" in item else item["q_id"]
            final_results.setdefault(q_id, [])
            final_results[q_id].append(item)
    return final_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--utility_pred', type=str, nargs="+")
    parser.add_argument('--retrieval_i_only', type=str,
                        default=None, nargs="+")
    parser.add_argument('--retrieval_multi', type=str, nargs="+")
    parser.add_argument('--groudness_pred', type=str, nargs="+")
    parser.add_argument('--relevance_pred', type=str, nargs="+")
    parser.add_argument('--orig_input_data', type=str, nargs="+")
    parser.add_argument('--retrieval_data', type=str, nargs="+")
    parser.add_argument('--splitted_input_data', type=str, nargs="+")
    parser.add_argument('--output_fn', type=str)
    parser.add_argument('--prev_result_fp', type=str)
    parser.add_argument('--negative_samples', action="store_true")
    args = parser.parse_args()

    decisions_need_retrieval_initial = load_all_files(args.retrieval_i_only)
    decisions_need_retrieval_multi = load_all_files(args.retrieval_multi)
    reward_utility = load_all_files(args.utility_pred)
    reward_relevance = load_all_files(args.relevance_pred)
    reward_groundness = load_all_files(args.groudness_pred)

    input_data = load_all_files(args.orig_input_data)
    splitted_output_data = load_all_files(args.splitted_input_data)
    retrieval_data = load_all_files(args.retrieval_data)

    processed_data = []
    need_retrieval_initial_stats = []
    need_retrieval_multi_stats = []
    utility_stats = []
    groundness_stats = []
    relevance_stats = []
    for q_id, instance in tqdm(input_data.items()):
        
        if q_id not in decisions_need_retrieval_initial:
            print("missing need retrieval")
            continue
        if q_id not in reward_utility:
            print("missing utility retrieval")
            continue
        
        dataset_name = instance[0]["dataset_name"]
        if args.negative_samples is True and dataset_name not in ["nq", "asqa", "fever", "wow", "arc_easy", "obqa"]:
            continue
        need_retrieval_initial_i = decisions_need_retrieval_initial[q_id][0]["pred"]
        reward_utility_i = reward_utility[q_id][0]["pred"]
        instruction = instance[0]["instruction"]
        output = instance[0]["output"]
        need_retrieval_i_token =  convert_score_to_retrieval_token(need_retrieval_initial_i)
        utility_i_token = convert_score_to_utility_token(reward_utility_i)
        if instance[0]["output"] in ["true", "false"] and utility_i_token == "[Utility:1]":
            utility_i_token =  "[Utility:5]"
        if utility_i_token is None:
            print("utility_i_token invalid")
            continue
        if need_retrieval_i_token is None:
            print("need_retrieval_i_token invalid")
            continue

        need_retrieval_initial_stats.append(need_retrieval_i_token)
        utility_stats.append(utility_i_token)
        if need_retrieval_initial_i == "[No Retrieval]":
            processed_output = need_retrieval_i_token + output + utility_i_token
            if random.random() > 0.6:
                need_retrieval_multi_stats.append(need_retrieval_initial_i)
                processed_data.append({"instruction": instruction, "output": processed_output, "input": "", "id": q_id, "dataset_name": dataset_name})
        else:
            if q_id not in decisions_need_retrieval_multi:
                print("missing need retrieval multi")
                print(q_id)
                continue
            decisions_need_retrieval_multi_i = decisions_need_retrieval_multi[q_id]
            decisions_need_retrieval_multi_i = {item["sent_idx"]: item["pred"] for item in decisions_need_retrieval_multi_i}
            splitted_output_sentences = splitted_output_data[q_id][0]["splitted_output"]
            skipped_sentences = splitted_output_data[q_id][0]["skipped"]
            retrieved_results = retrieval_data[q_id]
            retrieved_results = {item["sent_idx"]: item for item in retrieved_results}
            output = ""
            if q_id not in reward_relevance or q_id not in reward_groundness:
                print("{} not in relevance or groundness reward files.".format(q_id))
                continue
            for sent_idx, sentence in enumerate(splitted_output_sentences):
                if skipped_sentences[str(sent_idx)] is True:
                    output += sentence
                    continue
                if sent_idx not in decisions_need_retrieval_multi_i:
                    print("missing sent idx: {}".format(decisions_need_retrieval_multi_i.keys()))
                    continue
                if sent_idx == 0:
                    need_retrieval_decision_token = need_retrieval_i_token
                else:
                    need_retrieval_decision_token = convert_score_to_retrieval_token(decisions_need_retrieval_multi_i[sent_idx])
            
                if need_retrieval_decision_token == "[Retrieval]":
                    need_retrieval_decision_token == "[Retrieval]"
                    need_retrieval_multi_stats.append("[Retrieval]")
                    if q_id not in reward_relevance or q_id not in reward_groundness:
                        print("{} not in relevance or groundness reward files.".format(q_id))
                        continue
                    relevance_p = [item for item in reward_relevance[q_id] if item["sent_idx"] == sent_idx]
                    groundness_p = [item for item in reward_groundness[q_id] if item["sent_idx"] == sent_idx]
                    relevance_p_pidx = {item["p_idx"]: item for item in relevance_p}
                    groundness_p_pidx = {item["p_idx"]: item for item in groundness_p}

                    # Searching the best paragraph among retrieval results
                    fully_supported_p_indices = []
                    partially_supported_p_indices = []
                    no_support_p_indices = []
                    irrelevant_p_indices = []
                    for p_idx in range(len(relevance_p_pidx)):
                        relevance_token = postprocess_relevance_reward_token(relevance_p_pidx[p_idx]["pred"])
                        if p_idx not in groundness_p_pidx:
                            print("missing paragraph for p_index")
                            print(groundness_p_pidx.keys())
                            print(relevance_p_pidx.keys())
                            continue
                        groundness_token = convert_score_to_groudness(groundness_p_pidx[p_idx]["pred"])
                        if relevance_token is None or groundness_token is None:
                            continue
                        if relevance_token == "[Relevant]" and groundness_token ==  "[Fully supported]":
                            fully_supported_p_indices.append(p_idx)
                        elif relevance_token == "[Relevant]" and groundness_token == "[Partially supported]":
                            partially_supported_p_indices.append(p_idx)
                        elif relevance_token == "[Relevant]" and groundness_token == "[No support / Contradictory]":
                            no_support_p_indices.append(p_idx)
                        elif  relevance_token == "[Irrelevant]":
                            irrelevant_p_indices.append(p_idx)

                    # Form the output
                    if len(fully_supported_p_indices) > 0:
                        p_idx = fully_supported_p_indices[0]
                        if dataset_name == "nq" and random.random() > 0.75 and len(irrelevant_p_indices) > 0:
                            p_idx = random.sample(range(len(irrelevant_p_indices)), k=1)[0]
                    elif len(fully_supported_p_indices) == 0 and len(partially_supported_p_indices) > 0 and random.random() > 0.3:
                        p_idx = partially_supported_p_indices[0]
                    elif len(fully_supported_p_indices) == 0 and len(partially_supported_p_indices) == 0 and len(no_support_p_indices) > 0 and random.random() > 0.75:
                        p_idx = no_support_p_indices[0]
                    else:
                        if len(irrelevant_p_indices) == 0:
                            continue
                        else:
                            p_idx = random.sample(range(len(irrelevant_p_indices)), k=1)[0]

                    if args.negative_samples is True:
                        if len(irrelevant_p_indices) == 0:
                               continue
                        p_idx = random.sample(range(len(irrelevant_p_indices)), k=1)[0]

                    if dataset_name in ["nq", "fever"] and len(irrelevant_p_indices) > 0:
                        p_idx = random.sample(range(len(irrelevant_p_indices)), k=1)[0]

                    if p_idx not in relevance_p_pidx:
                        print("missing relevance for {}".format(p_idx))
                        continue
                    if p_idx not in groundness_p_pidx:
                        print("missing groudness for {}".format(p_idx))
                        continue
                    relevance_token = postprocess_relevance_reward_token(relevance_p_pidx[p_idx]["pred"])
                    groundness_token = convert_score_to_groudness(groundness_p_pidx[p_idx]["pred"])

                    if args.negative_samples is True and groundness_token == "[No support / Contradictory]":
                        relevance_token = "[Irrelevant]"
    
                    
                    if dataset_name in ["nq", "fever", "wow"] and groundness_token == "[No support / Contradictory]" and relevance_token == "[Relevant]":
                        relevance_token = "[Irrelevant]" 

                    if relevance_token is None or groundness_token is None:
                        continue
                    ctx = retrieved_results[sent_idx]["ctxs"][p_idx]
                    processed_paragraph = ctx["title"]+"\n" + ctx["text"]
                    if relevance_token == "[Relevant]":
                        output += "[Retrieval]" + "<paragraph>{}</paragraph>".format(processed_paragraph) + "[Relevant]" + sentence + groundness_token
                        groundness_stats.append(groundness_token)
                    else:
                        output += "[Retrieval]" + "<paragraph>{}</paragraph>".format(processed_paragraph) + "[Irrelevant]" + sentence
                    relevance_stats.append(relevance_token)
                elif need_retrieval_decision_token == "[Continue to Use Evidence]":
                    print("continue to retrieve")
                    need_retrieval_multi_stats.append(need_retrieval_decision_token)
                    output += "[Continue to Use Evidence]" + sentence
                elif need_retrieval_decision_token == "[No Retrieval]":
                    need_retrieval_multi_stats.append("[No Retrieval]")
                    output += "[No Retrieval]" + sentence

            if len(output) == 0:
                continue
            output += utility_i_token
            if (args.negative_samples is True and "[Irrelevant]" not in output) or (args.negative_samples is True  and "[No Retrieval]" in output):
                continue
            if dataset_name == "nq" and  "[No Retrieval]" in output and random.random() > 0.5: 
                continue
            if  len(postprocess(output)) > 0:
                processed_data.append({"instruction": instruction, "output": output, "input": "", "id": q_id, "dataset_name": dataset_name})
    
    for item in processed_data:
        if "REFUTES" in item["output"]:
            item["output"] = item["output"].replace("REFUTES", "false")
        if "SUPPORTS" in item["output"]:
            item["output"] = item["output"].replace("SUPPORTS", "true")

    print("overall stats")
    print(Counter(need_retrieval_initial_stats))
    print(Counter(need_retrieval_multi_stats))
    print(Counter(utility_stats))
    print(Counter(groundness_stats))
    print(Counter(relevance_stats))
    processed_data = [item for item in processed_data if item["output"].find("[Continue to Use Evidence]")!= 0]

    nq_sampled = random.sample([item for item in processed_data if item["dataset_name"] == "nq"], k=3)
    print("nq")
    for item in nq_sampled:
        print(item)

    nq_sampled = random.sample([item for item in processed_data if item["dataset_name"] == "asqa"], k=3)
    print("asqa")
    for item in nq_sampled:
        print(item)

    print(Counter([item["dataset_name"] for item in processed_data]))

    if args.prev_result_fp is not None:
        prev_result = load_jsonlines(args.prev_result_fp)
        print("prev data num: {}".format(len(prev_result)))
        prev_result = [item for item in prev_result if len(postprocess(item["output"])) > 0]
        print("prev data num: {}".format(len(prev_result)))
        processed_data = prev_result + processed_data
        print("final result num: {}".format(len(processed_data)))

    with open(args.output_fn + ".json", "w") as outfile:
        json.dump(processed_data, outfile)

    save_file_jsonl(processed_data, args.output_fn + ".jsonl")

if __name__ == "__main__":
    main()
