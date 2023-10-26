## Collecting Machine Generated Reward
This directory contains codes to generate machine-generated rewards for a given input-output pair. 
We use GPT-4 by default. 

## How to collect reward data
Follow README in [process_data](../process_data) to create the source data.

The input file is a `json` or `jsonl` file containing a list of entries. Each entry consists of 

```py
{
    "instruction": str, # input instruction 
    "target_output": str, # segment-level output
    "evidence": str, # retrieved Wikipedia paragraph
    "preceding_sentences": str, # previously generated sentences
    "output": str # full output (only used for utility),
    "q_id": str # unique instance id,
    "sent_id": int # sentence index
    "p_id": int # paragraph index
}
```

## How to collect reward data

We collect fine-grained feedback for the following four aspects. Use the following script to collect data. 
1. [Collect `IsUse`](chatgpt_utility.py)
2. [Collect retrieval tokens](chatgpt_need_retrieval.py)
3. [Collect `IsRel` Tokens](chatgpt_relevance.py)
3. [Collect `IsSup` tokens](chatgpt_groundness.py)

```sh
python chatgpt_utility.py \
    --input_file path_to_input_file \
    --output_file_name path_to_output_file \
    --api_key path_to_open_ai_api_txt_file \
    -org_name your_organization_name \
    --model_name open_ai_model_name \
```
