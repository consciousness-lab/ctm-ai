import json
import sys
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def parse_prediction(answer_text):
    """
    Parse the model's text response to determine the predicted label.
    Expected format: 'Yes...' or 'No...'
    Returns: True (Sarcasm) or False (Not Sarcasm), or None if unclear.
    """
    if not answer_text:
        return None
    
    # Normalize text
    text = answer_text.strip().lower()
    
    # Remove common prefixes if any (e.g. "Answer: Yes")
    if text.startswith("answer:"):
        text = text.replace("answer:", "").strip()
        
    # Check for Yes/No at the beginning
    if text.startswith('yes') or text.startswith('true'):
        return True
    elif text.startswith('no') or text.startswith('false'):
        return False
    
    # Fallback: Look for keywords if explicit Yes/No is missing (less reliable)
    # You might want to enable/disable this based on how strict you want to be
    if "is sarcastic" in text and "not sarcastic" not in text:
        return True
    if "not sarcastic" in text:
        return False
        
    return None

def calculate_metrics(file_path, verbose=False):
    y_true = []
    y_pred = []
    invalid_entries = 0
    total_entries = 0
    
    print(f"Processing file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                total_entries += 1
                try:
                    data = json.loads(line)
                    # The key is usually the file ID (e.g., "2_123"), so we iterate over keys
                    # Assuming one entry per line usually contains one test case
                    for test_id, content in data.items():
                        # Get ground truth
                        ground_truth = content.get('label', None)
                        if ground_truth is None:
                            print(f"[Warning] Line {line_num}: No label found for {test_id}")
                            continue
                        
                        # Get prediction
                        # 'answer' is usually a list, take the first one
                        answers = content.get('answer', [])
                        if not answers:
                             print(f"[Warning] Line {line_num}: No answer found for {test_id}")
                             continue
                        
                        model_response = answers[0]
                        prediction = parse_prediction(model_response)
                        
                        if prediction is None:
                            print(f"[Warning] Line {line_num}: Could not parse prediction for {test_id}. Response start: '{model_response[:50]}...'")
                            invalid_entries += 1
                            # Treat invalid as wrong? or skip? 
                            # Let's skip for metrics but report it
                            continue
                            
                        y_true.append(ground_truth)
                        y_pred.append(prediction)
                        
                        if verbose and ground_truth != prediction:
                             print(f"  [Mismatch] ID: {test_id} | True: {ground_truth} | Pred: {prediction}")

                except json.JSONDecodeError:
                    print(f"[Error] Line {line_num}: Invalid JSON")
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return

    if not y_true:
        print("No valid data found to calculate metrics.")
        return

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print("\n" + "="*30)
    print(f"Evaluation Results for: {file_path}")
    print("="*30)
    print(f"Total samples processed: {len(y_true)} (Skipped/Invalid: {invalid_entries})")
    print("-" * 20)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 20)
    print(f"Confusion Matrix:")
    print(f"TP: {tp} | FP: {fp}")
    print(f"FN: {fn} | TN: {tn}")
    print("="*30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate classification metrics from JSONL file.')
    parser.add_argument('file_path', type=str, help='Path to the .jsonl file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print details of mismatched predictions')
    
    args = parser.parse_args()
    
    calculate_metrics(args.file_path, args.verbose)

