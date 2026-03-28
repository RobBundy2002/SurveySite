import json
import glob
import sys
import numpy as np
from scipy import stats

# ── Condition mappings ────────────────────────────────────────────────────────

CONGRUENT = {
    "I finally got the research grant I've been stressing over! 🥂",
    "My laptop just blue-screened in the middle of a 2-hour exam. 💀",
    "The professor just gave me an extension without me even asking. 😇",
    "I just realized I've been muted for the last 10 minutes of my presentation. 🤡",
    "Seeing my name on the final graduation list feels so surreal. ✨",
    "I've had four cups of coffee and I still can't process this textbook. 😵\u200d💫",
    "Someone just left a huge dent in my car and didn't leave a note. 🤬",
    "The curve on the midterm was so generous that I actually passed. 😮\u200d💨",
    "I just found out my favorite professor is retiring this semester. 💔",
    "That feeling when you submit the final paper of your degree. 🌊",
}

CONTRADICTORY = {
    "My advisor hasn't responded to my emails in three weeks. 😌",
    "I've been rereading the same paragraph for the last hour. 😇",
    "I just found out my ex is in two of my classes next semester. 🤩",
    "My group stopped texting me after the disagreement and finished the project without me. 🤗",
    "I cried less today than yesterday so I think I'm getting over it! 😢",
    "I finally got the internship I actually wanted. 😐",
    "My professor pulled me aside to tell me my research proposal was exceptional. 😶",
    "I just found out I'm the only one in my friend group who got into the program. 😬",
    "I paid off my credit card for the first time since freshman year. 🙁",
    "My thesis defense got moved up because my committee thinks it's ready. 😰",
}

CONTROL = {
    "I passed all four of my finals and finished the semester with a 3.9 GPA.",
    "The professor pulled me aside to say my essay was the best she had read all year.",
    "I got into the graduate program I applied to on my first try.",
    "My research paper was accepted for publication in the department journal.",
    "I found out I received a full scholarship for my final year of school.",
    "I failed the midterm by fifteen points and it dropped my grade to a D.",
    "My laptop crashed the night before my thesis was due and I lost everything I had not backed up.",
    "I was dropped from the course due to an administrative error and lost my spot permanently.",
    "I studied for the wrong exam and had nothing to write for the first hour of the test.",
    "My group submitted the project without my section and I received a zero for the assignment.",
}

def get_condition(r):
    t = r.get("question_text", "")
    if t in CONGRUENT:     return "Congruent"
    if t in CONTRADICTORY: return "Contradictory"
    if t in CONTROL:       return "Control"
    return "Control" if r.get("correct_answer") is None else "Contradictory"

# ── Load files ────────────────────────────────────────────────────────────────

patterns = sys.argv[1:] if len(sys.argv) > 1 else ["*.json"]
files = []
for p in patterns:
    files.extend(glob.glob(p))
files = sorted(set(files))

if not files:
    print("No JSON files found. Usage: python report.py responses_*.json")
    sys.exit(1)

all_responses = []
participants  = []

for path in files:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pid = data["summary"]["participant_id"]
    for r in data["responses"]:
        r["condition"]      = get_condition(r)
        r["participant_id"] = pid
        all_responses.append(r)
    participants.append({
        "id":         pid,
        "summary":    data["summary"],
        "emoji_freq": data["responses"][0]["demographics"].get("emojiFrequency", "Unknown"),
        "age":        data["responses"][0]["demographics"].get("age", None),
    })

n_participants = len(participants)

# ── Helpers ───────────────────────────────────────────────────────────────────

def by_cond(cond):
    return [r for r in all_responses if r["condition"] == cond]

def acc(lst):
    s = [r for r in lst if r["is_correct"] is not None]
    return (sum(r["is_correct"] for r in s) / len(s) * 100) if s else 0

def mean_rt(lst):
    return np.mean([r["reaction_time_seconds"] for r in lst]) if lst else 0

def mean_conf(lst):
    return np.mean([r["confidence"] for r in lst]) if lst else 0

conds      = ["Congruent", "Contradictory", "Control"]
cond_lists = {c: by_cond(c) for c in conds}

def per_participant_acc(cond):
    out = []
    for p in participants:
        lst = [r for r in cond_lists[cond]
               if r["participant_id"] == p["id"] and r["is_correct"] is not None]
        out.append(acc(lst))
    return out

def per_participant_rt(cond):
    out = []
    for p in participants:
        lst = [r for r in cond_lists[cond] if r["participant_id"] == p["id"]]
        out.append(mean_rt(lst))
    return out

def per_item_stats(cond):
    items = {}
    for r in cond_lists[cond]:
        q = r["question_text"]
        if q not in items:
            items[q] = {"correct": 0, "incorrect": 0, "rt": [], "conf": []}
        if r["is_correct"] is True:
            items[q]["correct"] += 1
        elif r["is_correct"] is False:
            items[q]["incorrect"] += 1
        items[q]["rt"].append(r["reaction_time_seconds"])
        items[q]["conf"].append(r["confidence"])
    return items

cong_accs   = per_participant_acc("Congruent")
contra_accs = per_participant_acc("Contradictory")
ctrl_accs   = per_participant_acc("Control")
contra_errors = [r for r in cond_lists["Contradictory"] if r.get("is_correct") is False]
emoji_error_pct = len(contra_errors) / max(len(cond_lists["Contradictory"]), 1) * 100

# ── Report ────────────────────────────────────────────────────────────────────

SEP  = "=" * 70
HSEP = "-" * 70
lines = []

lines += [
    SEP,
    "  EMOJI SENTIMENT STUDY — AGGREGATE STATISTICAL REPORT",
    f"  N = {n_participants} participant(s)  |  {len(all_responses)} total responses",
    SEP,
]

# Section 1: Condition summary
lines += ["\n[ 1 ] CONDITION-LEVEL SUMMARY", HSEP]
lines.append(f"{'Condition':<18} {'Acc %':>7} {'Mean RT':>9} {'Mean Conf':>11} {'N resp':>8} {'Correct':>8} {'Wrong':>7}")
lines.append(HSEP)
for c in conds:
    lst     = cond_lists[c]
    scored  = [r for r in lst if r["is_correct"] is not None]
    correct = sum(1 for r in scored if r["is_correct"])
    wrong   = len(scored) - correct
    lines.append(
        f"{c:<18} {acc(lst):>6.1f}% {mean_rt(lst):>8.2f}s {mean_conf(lst):>10.2f}"
        f" {len(lst):>8} {correct:>8} {wrong:>7}"
    )
lines.append(HSEP)
overall_scored = [r for r in all_responses if r["is_correct"] is not None]
lines.append(
    f"{'OVERALL':<18} {acc(overall_scored):>6.1f}% {mean_rt(all_responses):>8.2f}s"
    f" {mean_conf(all_responses):>10.2f} {len(all_responses):>8}"
    f" {sum(1 for r in overall_scored if r['is_correct']):>8}"
    f" {sum(1 for r in overall_scored if not r['is_correct']):>7}"
)

# Section 2: Per-participant
lines += ["\n[ 2 ] PER-PARTICIPANT BREAKDOWN", HSEP]
lines.append(f"{'Participant':<14} {'Age':>4} {'EmojiFreq':<12} {'Cong%':>7} {'Contra%':>8} {'Ctrl%':>7} {'Overall%':>9} {'Calibration':>12}")
lines.append(HSEP)
for i, p in enumerate(participants):
    s = p["summary"]
    age_str = str(p['age']) if p['age'] is not None else '?'
    overall_pct = s.get('accuracy_percent') or 0
    calib = s.get('calibration_score') or 0
    lines.append(
        f"{p['id']:<14} {age_str:>4} {p['emoji_freq']:<12}"
        f" {cong_accs[i]:>6.1f}% {contra_accs[i]:>7.1f}% {ctrl_accs[i]:>6.1f}%"
        f" {overall_pct:>8.1f}%"
        f" {calib:>12.2f}"
    )

# Section 3: Inferential stats
lines += ["\n[ 3 ] INFERENTIAL STATISTICS", HSEP]
lines.append("  Accuracy (paired t-tests, per-participant means):")
if n_participants >= 3:
    for label, a, b in [
        ("Congruent vs Contradictory", cong_accs,  contra_accs),
        ("Control   vs Contradictory", ctrl_accs,  contra_accs),
        ("Congruent vs Control",       cong_accs,  ctrl_accs),
    ]:
        t, p_val = stats.ttest_rel(a, b)
        sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns")
        lines.append(f"    {label:<32}  t = {t:>7.3f}   p = {p_val:.4f}   {sig}")
else:
    lines.append("    (Fewer than 3 participants — skipped)")

lines.append("\n  Reaction time (paired t-tests, per-participant means):")
if n_participants >= 3:
    cong_rt   = per_participant_rt("Congruent")
    contra_rt = per_participant_rt("Contradictory")
    ctrl_rt   = per_participant_rt("Control")
    for label, a, b in [
        ("Congruent vs Contradictory", cong_rt,  contra_rt),
        ("Control   vs Contradictory", ctrl_rt,  contra_rt),
        ("Congruent vs Control",       cong_rt,  ctrl_rt),
    ]:
        t, p_val = stats.ttest_rel(a, b)
        sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns")
        lines.append(f"    {label:<32}  t = {t:>7.3f}   p = {p_val:.4f}   {sig}")
else:
    lines.append("    (Fewer than 3 participants — skipped)")

# Section 4: Contradictory error analysis
lines += ["\n[ 4 ] CONTRADICTORY CONDITION — ERROR ANALYSIS", HSEP]
lines += [
    f"  Total contradictory responses : {len(cond_lists['Contradictory'])}",
    f"  Errors (followed emoji)       : {len(contra_errors)} ({emoji_error_pct:.1f}%)",
    f"  Correct (followed text)       : {len(cond_lists['Contradictory']) - len(contra_errors)} ({100 - emoji_error_pct:.1f}%)",
    "\n  By participant:",
]
for p in participants:
    pid      = p["id"]
    p_contra = [r for r in cond_lists["Contradictory"]
                if r["participant_id"] == pid and r["is_correct"] is not None]
    p_errors = [r for r in p_contra if not r["is_correct"]]
    pct      = len(p_errors) / max(len(p_contra), 1) * 100
    lines.append(f"    {pid:<14} {len(p_errors):>2} errors / {len(p_contra):>2} scored  ({pct:.1f}%)")

# Section 5: Per-item contradictory
lines += ["\n[ 5 ] PER-ITEM STATS — CONTRADICTORY (sorted by error rate)", HSEP]
for q, d in sorted(per_item_stats("Contradictory").items(),
                   key=lambda x: x[1]["incorrect"], reverse=True):
    total   = d["correct"] + d["incorrect"]
    err_pct = d["incorrect"] / max(total, 1) * 100
    lines += [
        f"\n  Q: {q}",
        f"     Correct: {d['correct']}  |  Errors: {d['incorrect']}  |  Error rate: {err_pct:.1f}%",
        f"     Mean RT: {np.mean(d['rt']):.2f}s  |  Mean Confidence: {np.mean(d['conf']):.2f}",
    ]

# Section 6: Per-item congruent
lines += ["\n[ 6 ] PER-ITEM STATS — CONGRUENT (sorted by error rate)", HSEP]
for q, d in sorted(per_item_stats("Congruent").items(),
                   key=lambda x: x[1]["incorrect"], reverse=True):
    total   = d["correct"] + d["incorrect"]
    err_pct = d["incorrect"] / max(total, 1) * 100
    lines += [
        f"\n  Q: {q}",
        f"     Correct: {d['correct']}  |  Errors: {d['incorrect']}  |  Error rate: {err_pct:.1f}%",
        f"     Mean RT: {np.mean(d['rt']):.2f}s  |  Mean Confidence: {np.mean(d['conf']):.2f}",
    ]

# Section 7: Emoji frequency correlation
lines += ["\n[ 7 ] EMOJI FREQUENCY vs CONTRADICTORY ACCURACY", HSEP]
freq_map = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Always": 5}
xs, ys = [], []
for p, ca in zip(participants, contra_accs):
    fv = freq_map.get(p["emoji_freq"])
    if fv is not None:
        xs.append(fv); ys.append(ca)
if len(xs) >= 3:
    m, b, r, p_val, _ = stats.linregress(xs, ys)
    lines += [
        f"  r = {r:.3f}   p = {p_val:.4f}   slope = {m:.2f}   intercept = {b:.2f}",
        f"  Direction: {'positive' if r > 0 else 'negative'} correlation  "
        f"({'more' if r > 0 else 'less'} emoji use → "
        f"{'higher' if r > 0 else 'lower'} contradictory accuracy)",
    ]
else:
    lines.append("  (Not enough data points for regression)")

lines += ["\n" + SEP, "  END OF REPORT", SEP]

out_path = "aggregate_report.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Saved report → {out_path}")