import json

results = []
with open('results_v12_100.jsonl') as f:
    for line in f:
        d = json.loads(line)
        for k, v in d.items():
            v['id'] = k
            results.append(v)

fp = [r for r in results if r['label'] == 0 and r['pred'] == 1]
fn = [r for r in results if r['label'] == 1 and r['pred'] == 0]
parse_fail = [r for r in results if r['pred'] is None]
tp = [r for r in results if r['label'] == 1 and r['pred'] == 1]
tn = [r for r in results if r['label'] == 0 and r['pred'] == 0]

total = len(results)
print(f"Total: {total}")
print(f"TP={len(tp)} TN={len(tn)} FP={len(fp)} FN={len(fn)} ParseFail={len(parse_fail)}")
print(f"Accuracy: {(len(tp)+len(tn))/total*100:.1f}%")
print()

print("=== FALSE POSITIVES (label=0, pred=1) ===")
for r in fp:
    ans = r['answer'][:150]
    print(f"  {r['id']}: {ans}")
print()

print("=== FALSE NEGATIVES (label=1, pred=0) ===")
for r in fn:
    ans = r['answer'][:150]
    print(f"  {r['id']}: {ans}")
print()

print("=== PARSE FAILURES ===")
for r in parse_fail:
    ans = str(r.get('parsed_answer', ''))[:150]
    print(f"  {r['id']}: label={r['label']} parsed={ans}")
