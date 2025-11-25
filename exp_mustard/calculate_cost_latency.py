import json
import sys
import argparse
import statistics

# Pricing Constants (Same as in run_baseline_gemini.py)
COST_INPUT_PER_1M = 0.075
COST_OUTPUT_PER_1M = 0.30

def calculate_stats(file_path):
    latencies = []
    input_tokens = []
    output_tokens = []
    costs = []
    
    valid_entries = 0
    missing_entries = 0
    
    print(f"Processing file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    # Iterate over items (usually just one per line)
                    for test_id, content in data.items():
                        # Check if usage and latency data exists
                        usage = content.get('usage')
                        latency = content.get('latency')
                        
                        if usage is None or latency is None:
                            # Try to look for nested usage if structure differs, otherwise skip
                            missing_entries += 1
                            continue
                            
                        input_tok = usage.get('prompt_tokens', 0)
                        output_tok = usage.get('completion_tokens', 0)
                        
                        # Calculate Cost for this entry
                        cost = (input_tok / 1_000_000 * COST_INPUT_PER_1M) + \
                               (output_tok / 1_000_000 * COST_OUTPUT_PER_1M)
                        
                        latencies.append(latency)
                        input_tokens.append(input_tok)
                        output_tokens.append(output_tok)
                        costs.append(cost)
                        valid_entries += 1

                except json.JSONDecodeError:
                    print(f"[Error] Line {line_num}: Invalid JSON")
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return

    if valid_entries == 0:
        print("\n[Warning] No entries with 'usage' and 'latency' data found.")
        print(f"Total lines skipped due to missing data: {missing_entries}")
        print("Make sure the JSONL file was generated with the updated script that records stats.")
        return

    # Calculate Statistics
    avg_latency = statistics.mean(latencies)
    avg_input = statistics.mean(input_tokens)
    avg_output = statistics.mean(output_tokens)
    avg_cost = statistics.mean(costs)
    
    total_cost = sum(costs)
    total_latency = sum(latencies)

    print("\n" + "="*40)
    print(f"COST & LATENCY ANALYSIS: {file_path}")
    print("="*40)
    print(f"Valid Samples: {valid_entries}")
    print(f"Missing Data:  {missing_entries} (Skipped)")
    print("-" * 30)
    print(f"Average Latency:      {avg_latency:.2f} seconds")
    print(f"Average Input Tokens: {avg_input:.1f}")
    print(f"Average Output Tokens:{avg_output:.1f}")
    print(f"Average Cost:         ${avg_cost:.6f}")
    print("-" * 30)
    print(f"Total Cost (this file): ${total_cost:.6f}")
    print(f"Total Time (sum):       {total_latency:.2f} seconds")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate cost and latency statistics from JSONL file.')
    parser.add_argument('file_path', type=str, help='Path to the .jsonl file')
    
    args = parser.parse_args()
    
    calculate_stats(args.file_path)

