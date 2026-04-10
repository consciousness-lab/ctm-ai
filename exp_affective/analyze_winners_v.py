import json
import os
import sys

filename = sys.argv[1] if len(sys.argv) > 1 else 'results_v15_100.jsonl'
error_type = sys.argv[2] if len(sys.argv) > 2 else 'fp'

results = []
with open(filename) as f:
    for line in f:
        d = json.loads(line.strip())
        for k, v in d.items():
            v['id'] = k
            results.append(v)

if error_type == 'fp':
    errors = [r for r in results if r['label'] == 0 and r['pred'] == 1]
    print("=== FP winners ===")
elif error_type == 'fn':
    errors = [r for r in results if r['label'] == 1 and r['pred'] == 0]
    print("=== FN winners ===")
else:
    errors = results
    print("=== All ===")

lang_wins = 0
audio_wins = 0
video_wins = 0

for r in errors:
    sid = r['id']
    path = os.path.join('detailed_info', sid + '.json')
    if not os.path.exists(path):
        print("  %s: no log" % sid)
        continue
    with open(path) as f:
        log = json.load(f)
    iters = log.get('iterations', [])
    last = iters[-1] if iters else {}
    winner = last.get('winning_processor', '?')
    chunks = last.get('initial_phase', [])
    parts = []
    for c in chunks:
        name = c['processor_name'].replace('_processor', '')
        conf = c.get('confidence', 0)
        rel = c.get('relevance', 0)
        w = c.get('weight', 0)
        parts.append("%s:c=%.1f/w=%.2f" % (name, conf, w))
    wname = winner.replace('_processor', '')
    if 'language' in winner:
        lang_wins += 1
    elif 'audio' in winner:
        audio_wins += 1
    elif 'video' in winner:
        video_wins += 1
    print("  %s: winner=%-8s | %s" % (sid, wname, ", ".join(parts)))

print()
print("Summary: lang=%d, audio=%d, video=%d" % (lang_wins, audio_wins, video_wins))
