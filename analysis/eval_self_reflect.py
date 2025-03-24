import json
import os
import re

import fire
import numpy as np
import vllm
from oat_zero.qwen_math_eval_toolkit.grader import math_equal
from oat_zero.qwen_math_eval_toolkit.parser import \
    extract_answer as math_extract_answer
from tabulate import tabulate
from transformers import AutoTokenizer


def preprocess_box_response_for_qwen_prompt(sequence, answer):
    model_output = re.sub(
        r"^.*?<\|im_start\|>assistant",
        "<|im_start|>assistant",
        sequence,
        flags=re.DOTALL,
        count=1,
    )
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    extract_answer = math_extract_answer(model_output, data_name="math")

    if math_equal(prediction=extract_answer, reference=answer):
        box_match = 1.0
    else:
        box_match = -0.5

    if "boxed" not in model_output:
        box_match = -1.0

    return "", box_match


def preprocess_box_response_for_r1(sequence, answer):
    # detect the answer between <answer> </answer> tags
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", sequence)
    if match:
        extracted_answer = match.group(1)  # Assign extracted answer to a variable
    else:
        extracted_answer = None
    if extracted_answer is None:
        return "", -1
    else:
        if math_equal(prediction=extracted_answer, reference=answer):
            return "", 1
        else:
            return "", -0.5


def main(
    model_name: str = "Qwen/Qwen2.5-7B",  # Qwen/Qwen2.5-7B, Qwen/Qwen2.5-Math-1.5B, Qwen/Qwen2.5-Math-7B,
    # deepseek-ai/deepseek-math-7b-base, deepseek-ai/deepseek-math-7b-instruct,
    # meta-llama/Llama-3.1-8B,
    # microsoft/rho-math-7b-v0.1
    data_path: str = "./data/math_train_500.json",
    save_dir: str = "./output/self_reflect",
    temperature: float = 0.6,
    max_tokens: int = 1500,
    n: int = 500,  # num of test samples
    n_samples: int = 1,  # num of samples generated for each test sample
    run_tag: str = None,
):
    data = json.load(open(data_path))

    if temperature == 0:
        print(f"Using temperature 0, set n_samples to 1")
        n_samples = 1

    if run_tag is None:
        run_tag = model_name.replace("/", "_")

    n_data = min(n, len(data))
    file_name = f"{run_tag}_{n_data}_{n_samples}_{temperature}.json"
    file_path = f"{save_dir}/{file_name}"
    stats_path = file_path.replace(".json", "_stats.json")

    if os.path.exists(stats_path):
        print(f"Stats file {stats_path} exists. Skipping this model.")
        return

    # ========== step-1: run inference ========== #
    print(f"Step 1: run inference")

    if not os.path.exists(file_path):
        print(f"File {file_name} does not exist, running inference")

        if "deepseek" in model_name:
            print(f"Using DeepSeek model: {model_name}")
            model = vllm.LLM(model_name, swap_space=16)
        else:
            model = vllm.LLM(model_name)

        # Load tokenizer (replace with your specific model)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        sampling_params = vllm.SamplingParams(
            n=n_samples, temperature=temperature, top_p=0.9, max_tokens=max_tokens,
        )

        prompts = [x["input"] for x in data][:n_data]

        if "deepseek" in model_name:
            # send 5 prompts to the model at a time
            outputs = []
            batch_size = 10  # customize the batch_size
            for i in range(0, n_data, batch_size):
                batch_outputs = model.generate(
                    prompts[i : i + batch_size], sampling_params
                )
                outputs.extend(batch_outputs)
        else:
            # send all prompts
            outputs = model.generate(prompts, sampling_params)

        data = data[:n_data]
        for i, d in enumerate(data):
            for j in range(n_samples):
                d[f"output_{j}"] = outputs[i].outputs[j].text
                d[f"End_with_EOS_{j}"] = (
                    tokenizer.eos_token_id in outputs[i].outputs[j].token_ids
                )

        os.makedirs(save_dir, exist_ok=True)
        json.dump(
            data, open(file_path, "w"), indent=4,
        )
    else:
        print(f"File {file_name} exists. Skipping inference")

    # ========== step-2: eval metrics ========== #
    print(f"Step 2: get metrics")

    output = json.load(open(file_path))

    n_questions = n_data
    n_samples = n_samples

    print(
        f"Processing {n_questions} questions with {n_samples} responses for each question"
    )
    pass_ls = []
    reflection_ls = []
    rewards = []
    n_reflections = []
    count_correct_format_n_eos = 0  # measure instruction following ability
    for idx, o in enumerate(output):
        o["idx"] = idx
        is_corrects = []
        has_reflections = []

        for j in range(n_samples):
            if "r1" in file_name in file_name:
                _, r = preprocess_box_response_for_r1(o[f"output_{j}"], o["gt_answer"])
            else:
                _, r = preprocess_box_response_for_qwen_prompt(
                    o[f"output_{j}"], o["gt_answer"]
                )

            # keyword-based self-reflection detection
            keywords = {
                "recheck",
                "rethink",
                "reevaluate",
                "re-evaluate",
                "reevaluation",
                "re-examine",
                "reexamine",
                "try again",
                "check again",
                "think again",
                "go over the steps",
            }

            # Count occurrences of each reflection keyword in the output
            text = o[f"output_{j}"].lower()
            n_reflection = sum(text.count(word) for word in keywords)
            has_reflection = n_reflection > 0

            # check format and eos
            if r > -1 and o[f"End_with_EOS_{j}"] == 1:
                count_correct_format_n_eos += 1

            # binarize the reward
            if r == 1:
                is_correct = 1
            else:
                is_correct = 0

            has_reflections.append(has_reflection)
            is_corrects.append(is_correct)
            rewards.append(r)
            n_reflections.append(n_reflection)

        pass_ls.append(max(is_corrects))
        reflection_ls.append(max(has_reflections))

    # analyze the correlation between reward and reflection
    pass_ls = np.array(pass_ls)
    reflection_ls = np.array(reflection_ls)

    stats = {
        "n_prompts": len(output),
        "pass@k": int(pass_ls.sum()),
        "reflection@k": int(reflection_ls.sum()),
        "correct_format_eos_rate": round(
            count_correct_format_n_eos / (len(output) * n_samples), 3
        ),
    }

    # Print statistics in table format
    table = [[k, v] for k, v in stats.items()]
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))

    # Save statistics to json file
    stats.update({"rewards": rewards, "n_reflections": n_reflections})
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)


def collect_results(
    save_dir: str, match_pattern: str = ".*stats.json", title: str = None
):
    # collect all the results in the save_dir that match the pattern
    target_files = []
    for file in os.listdir(save_dir):
        if re.match(match_pattern, file):
            # print(f"Collecting {file}")
            target_files.append(file)

    # load the results
    results = []
    for file in target_files:
        results.append(json.load(open(os.path.join(save_dir, file))))

    # prepare table data
    exclude_keys = ["rewards", "n_reflections"]
    table_data = []
    metrics = [key for key in results[0].keys() if key not in exclude_keys]

    for file, result in zip(target_files, results):
        row = [file]
        for metric in metrics:
            row.append(result[metric])
        table_data.append(row)

    # print as CSV
    headers = ["Global Step"] + metrics
    if title:
        print(f"\n{title}")
    print(",".join(headers))  # Print header row
    for row in table_data:
        print(",".join(str(x) for x in row))  # Print data rows

    return


if __name__ == "__main__":
    fire.Fire({"main": main, "collect": collect_results})
