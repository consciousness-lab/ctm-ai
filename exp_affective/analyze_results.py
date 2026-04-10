import json
import sys

filename = sys.argv[1] if len(sys.argv) > 1 else 'results_v13_100.jsonl'

results = []
with open(filename) as f:
    for line in f:
        d = json.loads(line.strip())
        for k, v in d.items():
            v['id'] = k
            results.append(v)

fp = [r for r in results if r['label'] == 0 and r['pred'] == 1]
fn = [r for r in results if r['label'] == 1 and r['pred'] == 0]
parse_fail = [r for r in results if r['pred'] is None]
tp = [r for r in results if r['label'] == 1 and r['pred'] == 1]
tn = [r for r in results if r['label'] == 0 and r['pred'] == 0]

total = len(results)
pos = len([r for r in results if r['label'] == 1])
neg = len([r for r in results if r['label'] == 0])
print("File: %s" % filename)
print("Total: %d (pos=%d, neg=%d)" % (total, pos, neg))
print("TP=%d TN=%d FP=%d FN=%d ParseFail=%d" % (len(tp), len(tn), len(fp), len(fn), len(parse_fail)))
acc = (len(tp) + len(tn)) / total * 100
prec = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
rec = len(tp) / pos if pos > 0 else 0
neg_rec = len(tn) / neg if neg > 0 else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
print("Accuracy: %.1f%%" % acc)
print("Precision: %.3f  Pos Recall: %.3f  Neg Recall: %.3f  F1: %.3f" % (prec, rec, neg_rec, f1))
print()

print("=== FALSE POSITIVES (label=0, pred=1) [%d] ===" % len(fp))
for r in fp:
    ans = r['answer'][:150]
    print("  %s: %s" % (r['id'], ans))
print()

print("=== FALSE NEGATIVES (label=1, pred=0) [%d] ===" % len(fn))
for r in fn:
    ans = r['answer'][:150]
    print("  %s: %s" % (r['id'], ans))
print()

print("=== PARSE FAILURES [%d] ===" % len(parse_fail))
for r in parse_fail:
    ans = str(r.get('parsed_answer', ''))[:150]
    print("  %s: label=%s parsed=%s" % (r['id'], r['label'], ans))
