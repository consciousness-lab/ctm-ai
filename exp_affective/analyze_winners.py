import json
import os

fp_ids = ['7495','11010','4424','5906','10616','10374','877','13722','1745','9181','13459','2937','5907','9747','15547','14351']
fn_ids = ['1752','1328','11127','856','14773','3451','9835','3424','12388','14488','4889','11014','7168','2739','13231','7970','1578']

def analyze(sid_list, label):
    for sid in sid_list:
        path = os.path.join('detailed_info', sid + '.json')
        if not os.path.exists(path):
            print("  %s: no log" % sid)
            continue
        with open(path) as f:
            log = json.load(f)
        iters = log.get('iterations', [])
        last = iters[-1] if iters else {}
        winner = last.get('winning_processor', '?')
        weight = last.get('winning_weight', 0)
        chunks = last.get('initial_phase', [])
        parts = []
        for c in chunks:
            name = c['processor_name'].replace('_processor', '')
            conf = c.get('confidence', 0)
            rel = c.get('relevance', 0)
            w = c.get('weight', 0)
            parts.append("%s:c=%.2f/r=%.2f/w=%.2f" % (name, conf, rel, w))
        chunk_str = ", ".join(parts)
        print("  %s: winner=%s w=%.2f | %s" % (sid, winner, weight, chunk_str))

print("=== FP (label=0, pred=1): which processor won? ===")
analyze(fp_ids, 0)
print()
print("=== FN (label=1, pred=0): which processor won? ===")
analyze(fn_ids, 1)
