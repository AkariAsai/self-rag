## Create Generator Training Data

This repository consists of scripts to create training data. Our data creation for Generator consists of multiple steps. 

1. [process input data](#process-input-data)
2. [run Critic to judge retrieval tokens](#critic-evaluation-retrieval-tokens)
2. [run Critic to to judge isUse](#critic-evaluation-isuse)
3. [run initial Contriever](#run-contriever)
4. [Create isRel and isSUP data](#create-isrel-issup_ata)
4. [run Critic to judge isRel](#critic-evaluation-isrel)
5. [run Critic to judge isSup](#critic-evaluation-issup)
6. [Combine Data](#combine-data)

Note that the generator data creation involves many steps and takes time if you create more than 10k instances.  
We are working on adding improving the implementations and adding a single script. 
You can download our training data consisting of 150K instances [here](https://drive.google.com/file/d/10G_FozUV4u27EX0NjwVe-3YMUMeTwuLk/view?usp=share_link).


## Process Input data 

First we need to prepare the initial input data. Input data must follow the following schema. 

```
{"instruction": instruction, 
"output": output, 
"input": "",
"topic": "", 
"id": q_id, 
"dataset_name": dataset_name}
```
See examples script at [process_source_data](../process_data)


### Critic Evaluation Retrieval Tokens
To add predictions for retrieval necessity given the input data only, please run the command below. 

```
python run_reward_vllm.py \
    --input_file YOUR_INPUT_FILENAME \
    --model_name YOUR_CRITIC_MODEL_PATH \
    --task 'retrieval' \
    --inst_mode retrieval_instruction 
    --input_mode retrieval_input \
    --metric match \
    --result_fp INITIAL_RETRIEVAL_TOKEN_OUTPUT \
    --split test 
```

To add predictions for retrieval necessity given the input data and preceding sentences, please run the command below. Note that you need to create the initial retrieval file to evaluate the necessity of additional retrieval. 

```
python run_reward_vllm.py \
    --input_file YOUR_INPUT_FILENAME \
    --model_name YOUR_CRITIC_MODEL_PATH \
    --task 'multi_retrieval' \
    --inst_mode retrieval_multi_instruction \
    --input_mode retrieval_multi_input \
    --metric match \
    --result_fp MULTI_RETRIEVAL_TOKEN_OUTPUT \
    --split test 
```

### Critic Evaluation `isUse`
Run the command below. 
```
python run_reward_vllm.py \
    --input_file YOUR_INPUT_FILENAME \
    --model_name YOUR_CRITIC_MODEL_PATH \
    --task utility \
    --inst_mode utility_instruction \
    --input_mode utility_input \
    --result_fp UTILITY_OUTPUT \
    --split test 
```

# Run Contriever
## Initial retrieval
First preprocess data. 
```
python create_retrieval_data.py \
    --input_files INPUT_FILE \
    --output_file INITIAL_RETRIEVAL_INPUT
```
Then, run Contriever on the `INIT`

```
cd retrieval_lm
python passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco \
    --passages PATH_TO_CORPUS --passages_embeddings PATH_TO_EMBEDDINGS \
    --data INPUT_FILE \
    --output_dir INITIAL_RETRIEVAL_OUTPUT  --n_docs 10
```

## Continuous retrieval (`t>1`)
We only retrieve passages for the queries where initial retrieval necessity are `true`. First create the input file using `INITIAL_RETRIEVAL_TOKEN_OUTPUT` and `INITIAL_RETRIEVAL_OUTPUT`. 

```
python create_retrieval_data.py \
    --input_files INPUT_FILE \
    --output_file MULTI_RETRIEVAL_INPUT
    --need_retrieval_files INITIAL_RETRIEVAL_TOKEN_OUTPUT \
    --multiple_sent --initial_retrieval_file INITIAL_RETRIEVAL_OUTPUT
```

Then, run retrieval: 
```
cd retrieval_lm
python passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco \
    --passages PATH_TO_CORPUS --passages_embeddings PATH_TO_EMBEDDINGS \
    --data MULTI_RETRIEVAL_INPUT \
    --output_dir MULTI_RETRIEVAL_OUTPUT  --n_docs 10
```

## Create `IsRel` and `IsSUp` input data
Given the retrieval result from the previous step, create the input data for the `IsRel` and `IsSUp` tokens. 

As for `IsRel` and `IsSUp` requires large number of inferences, we recommend splitting data into multiple batch, by setting the `num_jobs` parameter. 
```
python create_prompt_data.py \
    --input_file MULTI_RETRIEVAL_OUTPUT \
    --output_dir YOUR_PROMPT_INPUT_DIR \
    --num_jobs NUM_JOBS \
    --top_n 10 \
    --multi_need_retrieval_pred_file MULTI_RETRIEVAL_TOKEN_OUTPUT
```

## Critic Evaluation `isRel`

```
python run_reward_vllm.py \
    --input_file YOUR_PROMPT_INPUT_DIR/prompt_data_batch_{BATCH_NUM}.jsonl  \
    --model_name YOUR_CRITIC_MODEL_PATH \
    --task 'relevance' --inst_mode relevance_instruction \
    --input_mode relevance_input \
    --metric match \
    --result_fp REL_OUTPUT_FILE_{BATCH_NUM} \
    --split test
```

## Critic Evaluation `isSup`

```
python run_reward_vllm.py \
    --input_file YOUR_PROMPT_INPUT_DIR/prompt_data_batch_{BATCH_NUM}.jsonl  \
    --model_name YOUR_CRITIC_MODEL_PATH \
    --task 'groudness' \
    --inst_mode ground_multi_instruction \
     --input_mode ground_multi_input 
    --metric match \
    --result_fp SUP_OUTPUT_FILE_{BATCH_NUM} \
    --split test
```

## Combine data
Finally you can combine the training data. 
```
python postprocess_data.py \
    --utility_pred YOUR_INPUT_FILENAME \
    --retrieval_i_only INITIAL_RETRIEVAL_TOKEN_OUTPUT \
    --retrieval_multi  MULTI_RETRIEVAL_TOKEN_OUTPUT \
    --groudness_pred ALL_SUP_OUTPUT_FILE \
    --relevance_pred ALL_REL_OUTPUT_FILE\
    --orig_input_data YOUR_INPUT_FILENAME \ 
    --retrieval_data MULTI_RETRIEVAL_OUTPUT \
    --splitted_input_data MULTI_RETRIEVAL_INPUT_splitted \
    --output_fn FINAL_OUTPUT_PATH
```