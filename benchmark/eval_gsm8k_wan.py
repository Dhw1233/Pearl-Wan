import os
import re
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import json
import time
import random
from src.engine_wan import PEARLWANEngine
from src.util_wan import parse_arguments, seed_everything

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
ANSWER_TRIGGER = "The answer is"


def create_demo_text(n_shot=8, cot_flag=True):
    question, chain, answer = [], [], []
    question.append("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?")
    chain.append("There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.")
    answer.append("6")
    question.append("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?")
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")
    question.append("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?")
    chain.append("Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.")
    answer.append("39")
    question.append("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?")
    chain.append("Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.")
    answer.append("8")
    question.append("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?")
    chain.append("Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.")
    answer.append("9")
    question.append("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?")
    chain.append("There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.")
    answer.append("29")
    question.append("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?")
    chain.append("Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.")
    answer.append("33")
    question.append("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?")
    chain.append("Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.")
    answer.append("8")

    index_list = list(range(len(question)))
    random.shuffle(index_list)
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
        else:
            demo_text += "Question: " + question[i] + "\nAnswer: " + ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def preprocess(input_text, prompt):
    return prompt + "Q: " + input_text + "\n" + "A:"


def postprocess(input_text, output_text, tokenizer):
    bos = tokenizer.bos_token or ""
    if bos and output_text.startswith(bos):
        generation = output_text[len(input_text)+len(bos)+1:]
    else:
        generation = output_text[len(input_text):]
    generation = generation.lower()
    generation = generation.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(generation) > 1 else False
    if answer_flag:
        pred = generation[1]
    else:
        pred = generation[-1]
    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]
    if len(pred) == 0:
        return INVALID_ANS
    if answer_flag:
        pred = pred[0]
    else:
        pred = pred[-1]
    if pred[-1] == ".":
        pred = pred[:-1]
    return pred


def load_data(data_path, limit=-1):
    prompt = create_demo_text()
    data = []
    with open(os.path.join(data_path, "gsm8k.jsonl")) as f:
        for i, line in enumerate(f.readlines()):
            if limit >= 0 and i >= limit:
                break
            datum = json.loads(line)
            datum["input_text"] = preprocess(datum["question"], prompt)
            datum["ground_truth"] = extract_answer_from_output(datum["answer"])
            data.append(datum)
    return data


def run_evaluation(args):
    print("="*70)
    print("PEARL-WAN GSM8K Evaluation")
    print("="*70)
    print(f"Draft Model: {args.draft_model}")
    print(f"Target Model: {args.target_model}")
    print(f"RTT: {args.rtt_ms}ms, Bandwidth: {args.bandwidth_mbps}Mbps")
    print(f"Limit: {args.limit}")
    print("="*70)

    engine = PEARLWANEngine(args)
    data = load_data(args.data_path, args.limit)

    results = {"config": vars(args), "runs": []}
    modes = ["wan"]
    if args.eval_mode == "wan":
        modes = ["autoregressive", "speculative_decoding", "wan"]

    for mode in modes:
        print(f"\n{'='*70}")
        print(f"Running mode: {mode}")
        print(f"{'='*70}")

        mode_results = []
        wall_times = {"time": [], "num_tokens": []}
        acc = 0

        for i, datum in enumerate(data):
            print(f"\n[Sample {i+1}/{len(data)}]")
            input_ids = engine.tokenizer.encode(datum["input_text"], return_tensors="pt")

            try:
                start = time.time()
                if mode == "autoregressive":
                    output_ids, _ = engine.autoregressive_sampling(input_ids)
                elif mode == "speculative_decoding":
                    output_ids, _ = engine.speculative_decoding_baseline(input_ids)
                else:
                    output_ids, _ = engine.pearl_wan_decode(input_ids)
                elapsed = time.time() - start

                new_tokens = output_ids.shape[1] - input_ids.shape[1]
                output_text = engine.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                answer = postprocess(datum["input_text"], output_text, engine.tokenizer)
                if answer == datum["ground_truth"]:
                    acc += 1

                speed = new_tokens / elapsed if elapsed > 0 else 0
                print(f"  Generated {new_tokens} tokens in {elapsed:.2f}s ({speed:.2f} tok/s) | Answer: {answer} (GT: {datum['ground_truth']})")

                if i != 0:
                    wall_times["time"].append(elapsed)
                    wall_times["num_tokens"].append(new_tokens)

                mode_results.append({
                    "question": datum["question"],
                    "new_tokens": new_tokens,
                    "time": elapsed,
                    "speed": speed,
                    "answer": answer,
                    "ground_truth": datum["ground_truth"],
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                mode_results.append({"question": datum["question"], "error": str(e)})

        avg_speed = sum(wall_times["num_tokens"]) / sum(wall_times["time"]) if wall_times["time"] else 0
        print(f"\n[Mode Summary: {mode}]")
        print(f"  Accuracy: {acc / len(data):.4f}")
        print(f"  Avg speed: {avg_speed:.2f} tok/s")
        if mode == "wan":
            engine.print_stats()

        results["runs"].append({
            "mode": mode,
            "avg_speed": avg_speed,
            "accuracy": acc / len(data) if data else 0,
            "details": mode_results,
        })

    out_path = os.path.join(args.exp_name, "eval_gsm8k_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    args = parse_arguments()
    run_evaluation(args)
