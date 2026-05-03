import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import json
import time
from src.engine_wan import PEARLWANEngine
from src.util_wan import parse_arguments


def load_data(data_path, limit=-1):
    data = []
    with open(os.path.join(data_path, "humaneval.jsonl")) as f:
        for i, line in enumerate(f.readlines()):
            if limit >= 0 and i >= limit:
                break
            datum = json.loads(line)
            datum["input_text"] = datum["prompt"].strip()
            data.append(datum)
    return data


def postprocess(tokenizer, input_text, output_text):
    bos = tokenizer.bos_token or ""
    if bos and output_text.startswith(bos):
        generation = output_text[len(input_text)+len(bos)+1:]
    else:
        generation = output_text[len(input_text):]
    stop_words = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", tokenizer.eos_token]
    for stop_word in stop_words:
        if stop_word in generation:
            next_line = generation.index(stop_word)
            generation = generation[:next_line].strip()
    output_text = input_text + '\n    ' + generation
    output_text = output_text.replace("\t", "    ")
    return output_text


def run_evaluation(args):
    print("="*70)
    print("PEARL-WAN HumanEval Evaluation")
    print("="*70)
    print(f"Draft Model: {args.draft_model}")
    print(f"Target Model: {args.target_model}")
    print(f"RTT: {args.rtt_ms}ms, Bandwidth: {args.bandwidth_mbps}Mbps")
    print(f"Adaptive Window: {args.enable_adaptive_window}")
    print(f"Compression: {args.enable_compression}")
    print(f"Fallback: {args.enable_fallback}")
    print(f"Limit: {args.limit}")
    print("="*70)

    engine = PEARLWANEngine(args)
    data = load_data(args.data_path, args.limit)

    results = {
        "config": vars(args),
        "runs": []
    }

    modes = ["wan"]
    if args.eval_mode == "wan":
        modes = ["autoregressive", "speculative_decoding", "wan"]

    for mode in modes:
        print(f"\n{'='*70}")
        print(f"Running mode: {mode}")
        print(f"{'='*70}")

        mode_results = []
        wall_times = {"time": [], "num_tokens": []}

        for i, datum in enumerate(data):
            print(f"\n[Sample {i+1}/{len(data)}] {datum['task_id']}")
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
                completion = postprocess(engine.tokenizer, datum["input_text"], output_text)

                speed = new_tokens / elapsed if elapsed > 0 else 0
                print(f"  Generated {new_tokens} tokens in {elapsed:.2f}s ({speed:.2f} tok/s)")

                if i != 0:
                    wall_times["time"].append(elapsed)
                    wall_times["num_tokens"].append(new_tokens)

                mode_results.append({
                    "task_id": datum["task_id"],
                    "prompt": datum["input_text"],
                    "new_tokens": new_tokens,
                    "time": elapsed,
                    "speed": speed,
                    "completion": completion,
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                mode_results.append({"task_id": datum["task_id"], "error": str(e)})

        avg_speed = sum(wall_times["num_tokens"]) / sum(wall_times["time"]) if wall_times["time"] else 0
        print(f"\n[Mode Summary: {mode}]")
        print(f"  Avg speed: {avg_speed:.2f} tok/s")
        if mode == "wan":
            engine.print_stats()

        results["runs"].append({
            "mode": mode,
            "avg_speed": avg_speed,
            "details": mode_results,
        })

    out_path = os.path.join(args.exp_name, "eval_humaneval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    args = parse_arguments()
    run_evaluation(args)
