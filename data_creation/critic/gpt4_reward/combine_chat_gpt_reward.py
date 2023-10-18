import json
import jsonlines
import argparse
import random
from collections import Counter

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

# Done
def create_utility_data(input_data):
    print("creating retrieval data")
    processed_data = []
    for item in input_data:
        input = item["input"]
        raw_output = item["raw_output"]
        if item["score"] == "":
            item["score"] = raw_output.split("\n")[0]
        output = item["score"]
        if output not in [1,2,3,4,5] or len(str(output)) == 0:
            continue
        label = "[Utility:{}]".format(output)
        processed_data.append({"instruction": PROMPT_DICT["utility_instruction"], "input": PROMPT_DICT["utility_input"].format_map(input), "output": label, "task": "utility"})
    print(processed_data[-1])
    print("total data number: {}".format(len(processed_data)))
    print(Counter([item["output"] for item in processed_data ]))
    return processed_data

# Done
def create_retrieval_data(input_data, multi_retrieval=False):
    print("creating multi sentence retrieval data")
    processed_data = []
    for item in input_data:
        input = item["input"]
        output = item["decision_token"]

        if len(str(output)) == 0:
            continue
        if output not in [ "[Retrieval]", "[No Retrieval]", "[Continue to Use Evidence]"]:
            continue

        if "sent_idx" not in item or item["sent_idx"] == 0 or len(item["preceding_sentences"]) == 0:
            processed_data.append({"instruction": PROMPT_DICT["multi_retrieval_three_way_instruction"], "input": PROMPT_DICT["multi_retrieval_three_way_input_wo_preceding"].format_map(input), "output": output, "task": "multi_retrieval"})
        else:
            processed_data.append({"instruction": PROMPT_DICT["multi_retrieval_three_way_instruction"], "input": PROMPT_DICT["multi_retrieval_three_way_input"].format_map(input), "output": output, "task": "retrieval"})
    print(processed_data[-1])
    print("total data number: {}".format(len(processed_data)))
    print(Counter([item["output"] for item in processed_data ]))
    return processed_data

# Done
def create_retrieval_data_input_only(input_data):
    print("creating retrieval data")
    processed_data = []
    for item in input_data:
        input = {"instruction": item["input"].split("##\nTask instruction: ")[1]}
        output = item["output"]
        if len(str(output)) == 0:
            continue
        if "Yes" in output:
            output = "[Retrieval]"
        elif "No" in output:
            output = "[No Retrieval]"
        else:
            continue
        assert output in [ "[Retrieval]", "[No Retrieval]" ]

        processed_data.append({"instruction": PROMPT_DICT["retrieval_instruction"], "input": PROMPT_DICT["retrieval_input"].format_map(input), "output": output, "task": "retrieval"})
    print(processed_data[-1])
    print("total data number: {}".format(len(processed_data)))
    print(Counter([item["output"] for item in processed_data ]))
    return processed_data

# Done
def create_groundness_data(input_data, multi_retrieval=False):
    print("creating groundness data")
    processed_data = []
    for item in input_data:
        input = item["input"]
        raw_output = item["raw_output"]
        if item["score"] == "":
            item["score"] = raw_output.split("\n")[0]
        if len(item["score"]) > 0 and item["score"][-1] == " ":
            item["score"] = item["score"][:-1]
        if len(item["score"]) == 0 or item["score"] not in ["[No support / Contradictory]", "[Fully supported]", "[Partially supported]"]:
            continue
        if multi_retrieval is True:
            if "sent_idx" not in item or item["sent_idx"] == 0 or len(item["preceding_sentences"]) == 0:
                processed_data.append({"instruction": PROMPT_DICT["ground_multi_instruction"], "input": PROMPT_DICT["ground_multi_input_wo_preceding"].format_map(input), "output": item["score"], "task": "groudness"})
            else:
                processed_data.append({"instruction": PROMPT_DICT["ground_multi_instruction"], "input": PROMPT_DICT["ground_multi_input"].format_map(input), "output": item["score"], "task": "groudness"})
        else: 
            processed_data.append({"instruction": PROMPT_DICT["ground_instruction"], "input": PROMPT_DICT["ground_input"].format_map(input), "output": item["score"], "task": "groudness"})
    print(processed_data[-1])
    print("total data number: {}".format(len(processed_data)))
    print(Counter([item["output"] for item in processed_data ]))
    return processed_data

# Done
def create_relevance_data(input_data):
    print("creating relevance data")
    processed_data = []
    for item in input_data:
        input = item["input"]
        raw_output = item["raw_output"]
        if item["score"] == "":
            item["score"] = raw_output.split("\n")[0]
        if len(item["score"]) > 0 and item["score"][-1] == " ":
            item["score"] = item["score"][:-1]
        if item["score"] not in ["[Relevant]", "[Irrelevant]"]:
            continue
        label = item["score"]
        if label == "[Relevant]" and random.random() > 0.7:
            continue
        processed_data.append({"instruction": PROMPT_DICT["relevance_instruction"], "input": PROMPT_DICT["relevance_input"].format_map(input), "output": label, "task": "relevance"})
    print(processed_data[-1])
    print("total data number: {}".format(len(processed_data)))
    print(Counter([item["output"] for item in processed_data ]))
    return processed_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ut_file', type=str, nargs="+")
    parser.add_argument('--ret_file', type=str, nargs="+")
    parser.add_argument('--multi_ret_file', type=str, nargs="+")
    # parser.add_argument('--ground_file', type=str,  nargs="+")
    parser.add_argument('--multi_ground_file', type=str,  nargs="+")
    parser.add_argument('--rel_file', type=str, nargs="+")
    # parser.add_argument('--multi_retrieval', action="store_true")
    parser.add_argument('--output_file_name', type=str)
    args = parser.parse_args()

    ret_source_data = []
    rel_source_data = []
    multi_ret_source_data = []
    multi_ground_source_data = []
    gd_source_data = []
    ut_source_data = []
    for fn in args.ret_file:
        for item in json.load(open(fn)):
            ret_source_data.append(item)
    for fn in args.multi_ret_file:
        for item in json.load(open(fn)):
            multi_ret_source_data.append(item)

    for fn in args.rel_file:
        for item in json.load(open(fn)):
            rel_source_data.append(item)
    for fn in args.multi_ground_file:
        for item in json.load(open(fn)):
            multi_ground_source_data.append(item)

    for fn in args.ut_file:
        for item in json.load(open(fn)):
            ut_source_data.append(item)

    final_train_data = []
    final_train_data += create_utility_data(ut_source_data)
    # final_train_data += create_retrieval_data(ret_source_data)
    final_train_data += create_retrieval_data_input_only(ret_source_data)
    final_train_data += create_retrieval_data(multi_ret_source_data)
    final_train_data += create_relevance_data(rel_source_data)
    final_train_data += create_groundness_data(multi_ground_source_data, True)

    final_train_data = [item for item in final_train_data if "### Response:" not in item["input"] and "### Response:" not in item["instruction"]]
    random.shuffle(final_train_data)
    train_data = final_train_data[1500:]
    dev_data = final_train_data[:1500]

    with open(args.output_file_name + "_train.json", "w") as outfile:
        json.dump(train_data, outfile)
    with open(args.output_file_name + "_dev.json", "w") as outfile:
        json.dump(dev_data, outfile)

if __name__ == "__main__":
    main()
