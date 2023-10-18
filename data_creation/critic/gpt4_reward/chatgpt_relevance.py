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


KNOWLEDGE_INSTRUCTIONS = {"nq": "Please answer the following questions using the shortest possible response. For example, if the question asks 'What is the capital of France?'', you can simply reply with 'Paris'.",
                          "fever": "Determine whether the following statement is true or false.",
                          "wow": "You have been provided with a chat history between two agents, separated by new lines. Generate a response that is informative and engaging based on the latest message."}

PROMPT_DICT = {
    "multi": (
        "You'll be provided with an instruction, along with evidence and possibly some preceding sentences. "
        "When there are preceding sentences, your focus should be on the sentence that comes after them. "
        "Your job is to determine if the evidence is relevant to the initial instruction and the preceding context, and provides useful information to complete the task described in the instruction. "
        "If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].\n\n"
        "###\nInstruction: Given four answer options, A, B, C, and D, choose the best answer.\n\n"
        "Input: Earth rotating causes\n"
        "A: the cycling of AM and PM\nB: the creation of volcanic eruptions\nC: the cycling of the tides\nD: the creation of gravity\n\n"
        "Evidence: Rotation causes the day-night cycle which also creates a corresponding cycle of temperature and humidity creates a corresponding cycle of temperature and humidity. Sea level rises and falls twice a day as the earth rotates.\n\n"
        "Rating: [Relevant]\n"
        "Explanation: The evidence explicitly mentions that the rotation causes a day-night cycle, as described in the answer option A.\n\n"
        "###\nInstruction: age to run for us house of representatives\n\n"
        "Evidence: The Constitution sets three qualifications for service in the U.S. Senate: age (at least thirty years of age); U.S. citizenship (at least nine years); and residency in the state a senator represents at the time of election.\n\n"
        "Rating: [Irrelevant]\n"
        "Explanation: The evidence only discusses the ages to run for the US Senate, not for the House of Representatives.\n\n"
        "###\nInstruction: {instruction}\n\n"
        "Evidence: {evidence}\n\n"
        "Rating:"
    ),
    "multi_no_preceding": (
        "You'll be provided with an instruction, along with evidence and possibly some preceding sentences. "
        "When there are preceding sentences, your focus should be on the sentence that comes after them. "
        "Your job is to determine if the evidence is relevant to the initial instruction and the preceding context, and provides useful information to complete the task described in the instruction. "
        "If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].\n\n"
        "###\nInstruction: Given four answer options, A, B, C, and D, choose the best answer.\n\n"
        "Input: Earth rotating causes\n"
        "A: the cycling of AM and PM\nB: the creation of volcanic eruptions\nC: the cycling of the tides\nD: the creation of gravity\n\n"
        "Evidence: Rotation causes the day-night cycle which also creates a corresponding cycle of temperature and humidity creates a corresponding cycle of temperature and humidity. Sea level rises and falls twice a day as the earth rotates.\n\n"
        "Rating: [Relevant]\n"
        "Explanation: The evidence explicitly mentions that the rotation causes a day-night cycle, as described in the answer option A.\n\n"
        "###\nInstruction: Describe a leader or a politician whom you admire. \n\n"
        "Preceding sentences: Leaders and politicians have the power to shape the course of history and impact the lives of countless individuals. Among the myriad of notable figures, Nelson Mandela stands as an exemplary leader whose indomitable spirit, unwavering commitment to justice, and remarkable ability to unite a divided nation have made him an admired and revered personality on a global scale. "
        "Evidence: Barack Obama was one of the most influential people of the world and the man with a difference. He has served as the President of the United States of America. He was the 44th President of America. He was elected in the year 2009 to the office of the President. He was the first-ever African-American President of America.\n\n"
        "Rating: [Irrelevant]\n"
        "Explanation: While the evidence discuss Barack Obama, who is known as an influential political leader, the preceding sentences describe Nelson Mandela, so this evidence doesn't provide useful information to generate an helpful continuation.\n\n"
        "###\nInstruction: {instruction}\n\n"
        "Evidence: {evidence}\n\n"
        "Rating:"
    ),
}


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def postprocess(results):
    raw_output = results["choices"][0]["message"]["content"]
    print(raw_output)
    if "\nExplanation:" in raw_output:
        explanation = raw_output.split("\nExplanation:")[1]
        if explanation[0] == " ":
            explanation = explanation[1:]
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


def process_input(example, multi_retrieval=False):
    if multi_retrieval is False:
        return PROMPT_DICT["context"].format_map(example)
    else:
        if "sent_idx" not in example or example["sent_idx"] == 0 or len(example["preceding_sentences"]) == 0:
            return PROMPT_DICT["multi_no_preceding"].format_map(example)
        else:
            return PROMPT_DICT["multi"].format_map(example)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, nargs='+')
    parser.add_argument('--output_file_name', type=str)
    parser.add_argument('--multi_retrieval', action="store_true")
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--org_name', type=str)
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo")
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

    result_list = []
    if args.n is not None and len(examples) > args.n:
        examples = random.sample(examples, k=args.n)

    task_types = Counter([item["dataset_name"]
                         for item in examples if "dataset_name" in item])

    print(Counter(task_types))

    for idx, example in tqdm(enumerate(examples)):
        if "output" not in example and "answers" in example:
            example["output"] = example["answers"][0] if type(
                example["answers"]) is list else example["answers"]
        if "target_output" not in example and "output" in example:
            example["target_output"] = example["output"]
        if "instruction" not in example and "question" in example:
            data_type = example["q_id"].split("_")[0]
            if data_type in KNOWLEDGE_INSTRUCTIONS:
                example["instruction"] = KNOWLEDGE_INSTRUCTIONS[data_type] + \
                    example["question"]
            else:
                example["instruction"] = example["question"]
        if "As a language model, I cannot" in example["output"]:
            continue
        input = process_input(example, multi_retrieval=args.multi_retrieval)
        try:
            results = completions_with_backoff(
                model=args.model_name,
                messages=[
                    {"role": "user",
                        "content": input},
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
                print("Evidence: {}".format(example["evidence"]))
                print("Score: {0} ({1})".format(score, explanation))

        except (APIError, Timeout, APIConnectionError):
            results = "ERROR: API error outputs"
        if idx % 20 == 0:
            print("saved output at {}".format(args.output_file_name + "_tmp"))
            with open(args.output_file_name + "_tmp", "w") as outfile:
                json.dump(result_list, outfile)

    with open(args.output_file_name, "w") as outfile:
        json.dump(result_list, outfile)


if __name__ == "__main__":
    main()
