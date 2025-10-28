from argparse import ArgumentParser
from pathlib import Path

import torch
import contexttimer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from peft import PeftModel
from util import read_json, save_json
from efficienedit.utils import Timer
from efficienedit.speculative_sampling import autoregressive_sampling,speculative_sampling_original,efficient_edit_speculative_sampling, autoregressive_sampling2

from icecream import install
install()

torch.manual_seed(520)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--approach", type=str, choices=["AR", "SD", "EE"], default="AR")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument('--output_dir', type=Path, default="result")
    parser.add_argument('--data_file', type=Path, default="data/CanItEdit/test.jsonl")
    parser.add_argument('--draft_lora_path', type=Path, default=None)
    parser.add_argument('--draft_model', type=Path, default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument('--target_model', type=Path, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    return parser.parse_args()

def code_edit_prompt(instruction,code_before):
    prompt = f"""###User:
You are an expert code editor. Please modify the given ##Code File according to the ##Instruction provided. Provide the complete revised code file with all modifications implemented.
##Instruction
{instruction}
##Code File
```python
{code_before}
```
###Assistant
##Code File
```python
"""
    return prompt

def speculative_sampling_inference(target_model, draft_model, eos_token_id_tensor, input_ids , max_token = 4096, temperature= 0.2, top_p = 0.95,top_k = 5):
    with torch.no_grad():
        outputs, drafter_timer, target_timer = speculative_sampling_original(
            prefix = input_ids,
            approx_model = draft_model,
            target_model = target_model,
            eos_token_id_tensor = eos_token_id_tensor,
            max_len=1500,
            temperature = temperature,
            top_k= top_k,
            top_p = top_p,
            )
    num_generated_tokens = outputs.shape[-1] - input_ids.shape[-1]
    completions = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=False)
    drafter_timer_dict, target_timer_dict = {f"{k}_drafter": v for k, v in drafter_timer.to_dict().items()}, {f"{k}_target": v for k, v in target_timer.to_dict().items()}
    total_dict = {f"total_{name}": drafter_timer_dict[f"{name}_drafter"] + target_timer_dict[f"{name}_target"] for name in ["n_tokens", "throughput", "time"]}
    ret = drafter_timer_dict | target_timer_dict | total_dict | {"num_generated_tokens":num_generated_tokens}
    return {"stats": dict(sorted(ret.items()))} | {"completions": completions}

def efficient_edit_inference(target_model, draft_model, code_before, eos_token_id_tensor, input_ids , max_token = 4096, temperature= 0.2, top_p = 0.95,top_k = 5):
    precode = tokenizer.encode(code_before, add_special_tokens=False, return_tensors="pt").to(target_model.device)
    with contexttimer.Timer() as t:
        with torch.no_grad():
            outputs = efficient_edit_speculative_sampling(
                prefix = input_ids,
                precode = precode,
                target_model = target_model,
                draft_model = draft_model,
                eos_token_id_tensor = eos_token_id_tensor,
                max_len=1500 ,
                policy = "greedy",
                temperature = temperature,
                top_k= top_k,
                top_p = top_p)
    time = t.elapsed
    tokens = outputs.shape[-1] - input_ids.shape[-1]
    result = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return {"time":time, "tokens":tokens, "throughput": tokens/time,"result":result}

def autoregressive_inference(model, input_ids, max_token, eos_token_id_tensor, temperature= 0.2, top_p = 0.95,top_k = 5):
    with torch.no_grad():
        outputs, target_timer = autoregressive_sampling2(input_ids, model, eos_token_id_tensor,  max_token, temperature, top_k, top_p)
    num_generated_tokens = outputs.shape[-1] - input_ids.shape[-1]
    completions = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=False)
    target_timer_dict = {f"{k}_target": v for k, v in target_timer.to_dict().items()}
    ret = target_timer_dict | {"num_generated_tokens":num_generated_tokens}
    return {"stats": dict(sorted(ret.items()))} | {"completions": completions}

if __name__ == '__main__':
    args = get_parser()

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model,torch_dtype=torch.float16,device_map="auto")
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model, torch_dtype=torch.float16,device_map="auto")

    if 'Qwen' in str(args.target_model):
        # these tokens are things like ``\n, ```, ```\\n, etc.
        eos_token_id_tensor = torch.tensor([84274,73594,9902,13874,41233,54275,151645]).to(target_model.device)
    else:
        eos_token_id_tensor = torch.tensor([10252,32021]).to(target_model.device)

    if args.draft_lora_path:
        print("load lora")
        draft_model = PeftModel.from_pretrained(draft_model, args.draft_lora_path)
        draft_model = draft_model.merge_and_unload()
        print("load lora end")
    draft_model.eval()
    target_model.eval()

    data = read_json(args.data_file,False)
    result = []

    if args.num_samples: data = data[:args.num_samples]
    for item in tqdm(data):
        ####canitedit####
        prompt = code_edit_prompt(item['instruction_lazy'],item['before'])
        code_before = item['before']+'\n'
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(target_model.device)

        if args.approach == "AR":
            output = autoregressive_inference(target_model, input_ids,2048, eos_token_id_tensor, temperature=0)
        elif args.approach == "SD":
            output = speculative_sampling_inference(target_model = target_model,draft_model = draft_model, eos_token_id_tensor = eos_token_id_tensor, input_ids = input_ids,max_token = 4096, temperature=0)
        elif args.approach == "EE":
            output = efficient_edit_inference(target_model = target_model,draft_model = draft_model, code_before= code_before, eos_token_id_tensor = eos_token_id_tensor, input_ids = input_ids,max_token = 4096, temperature=0)
        else:
            raise ValueError(f"invalid {args.approach=}")
        ic(output)

        # item['draft_rate'] = output['draft_rate']
        if 'edit_eval' in str(args.data_file):
            item['output'] = [output['result']]
        result.append(item | output)
        output_path = args.output_dir / f"result_{args.approach}.jsonl"
        save_json(output_path,result)
