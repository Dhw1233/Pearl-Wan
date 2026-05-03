import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import json
import time
from src.engine_wan import PEARLWANEngine
from src.util_wan import parse_arguments

def create_test_data():
    """Create simple test prompts if no data file exists."""
    prompts = [
        "def fibonacci(n):",
        "Write a Python function to sort a list of integers using quicksort.",
        "Explain the concept of speculative decoding in large language models.",
        "What is the difference between cloud computing and edge computing?",
        "Implement a function to check if a string is a palindrome.",
    ]
    return prompts

def run_evaluation(args):
    print("="*70)
    print("PEARL-WAN Evaluation")
    print("="*70)
    print(f"Draft Model: {args.draft_model}")
    print(f"Target Model: {args.target_model}")
    print(f"RTT: {args.rtt_ms}ms, Bandwidth: {args.bandwidth_mbps}Mbps")
    print(f"Adaptive Window: {args.enable_adaptive_window}")
    print(f"Compression: {args.enable_compression}")
    print(f"Fallback: {args.enable_fallback}")
    print("="*70)
    
    engine = PEARLWANEngine(args)
    prompts = create_test_data()
    
    results = {
        "config": vars(args),
        "runs": []
    }
    
    # Test different modes
    modes = ["wan"]
    if args.eval_mode == "wan":
        # Also run baseline for comparison
        modes = ["autoregressive", "speculative_decoding", "wan"]
    
    for mode in modes:
        print(f"\n{'='*70}")
        print(f"Running mode: {mode}")
        print(f"{'='*70}")
        
        mode_results = []
        total_tokens = 0
        total_time = 0
        
        for i, prompt in enumerate(prompts[:args.num_samples]):
            print(f"\n[Prompt {i+1}/{min(args.num_samples, len(prompts))}]")
            print(f"Text: {prompt[:80]}...")
            
            input_ids = engine.tokenizer.encode(prompt, return_tensors="pt")
            
            try:
                if mode == "autoregressive":
                    output_ids, elapsed = engine.autoregressive_sampling(input_ids)
                elif mode == "speculative_decoding":
                    output_ids, elapsed = engine.speculative_decoding_baseline(input_ids)
                else:  # wan
                    output_ids, elapsed = engine.pearl_wan_decode(input_ids)
                
                generated_text = engine.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                new_tokens = output_ids.shape[1] - input_ids.shape[1]
                speed = new_tokens / elapsed if elapsed > 0 else 0
                
                print(f"  Generated {new_tokens} tokens in {elapsed:.2f}s ({speed:.2f} tok/s)")
                
                mode_results.append({
                    "prompt": prompt,
                    "new_tokens": new_tokens,
                    "time": elapsed,
                    "speed": speed,
                    "output": generated_text[:200],
                })
                total_tokens += new_tokens
                total_time += elapsed
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                mode_results.append({
                    "prompt": prompt,
                    "error": str(e),
                })
        
        avg_speed = total_tokens / total_time if total_time > 0 else 0
        print(f"\n[Mode Summary: {mode}]")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average speed: {avg_speed:.2f} tok/s")
        
        if mode == "wan":
            engine.print_stats()
        
        results["runs"].append({
            "mode": mode,
            "avg_speed": avg_speed,
            "total_tokens": total_tokens,
            "total_time": total_time,
            "details": mode_results,
        })
    
    # Save results
    out_path = os.path.join(args.exp_name, "eval_wan_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    args = parse_arguments()
    run_evaluation(args)
