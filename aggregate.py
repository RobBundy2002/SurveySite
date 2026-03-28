import json
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Condition mappings (sourced from study HTML) ──────────────────────────────

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

CONTROL_POSITIVE = {
    "I passed all four of my finals and finished the semester with a 3.9 GPA.",
    "The professor pulled me aside to say my essay was the best she had read all year.",
    "I got into the graduate program I applied to on my first try.",
    "My research paper was accepted for publication in the department journal.",
    "I found out I received a full scholarship for my final year of school.",
}

CONTROL_NEGATIVE = {
    "I failed the midterm by fifteen points and it dropped my grade to a D.",
    "My laptop crashed the night before my thesis was due and I lost everything I had not backed up.",
    "I was dropped from the course due to an administrative error and lost my spot permanently.",
    "I studied for the wrong exam and had nothing to write for the first hour of the test.",
    "My group submitted the project without my section and I received a zero for the assignment.",
}

CONTROL_ANSWERS = {q: "positive" for q in CONTROL_POSITIVE}
CONTROL_ANSWERS.update({q: "negative" for q in CONTROL_NEGATIVE})

def get_condition(r):
    t = r.get("question_text", "")
    if t in CONGRUENT:      return "Congruent"
    if t in CONTRADICTORY:  return "Contradictory"
    if t in CONTROL:        return "Control"
    # fallback: use correct_answer nullness
    return "Control" if r.get("correct_answer") is None else "Contradictory"

# ── Load files ────────────────────────────────────────────────────────────────

patterns = sys.argv[1:] if len(sys.argv) > 1 else ["*.json"]
files = []
for p in patterns:
    files.extend(glob.glob(p))
files = sorted(set(files))

if not files:
    print("No JSON files found. Usage: python aggregate.py responses_*.json")
    sys.exit(1)

all_responses = []
participants  = []

for path in files:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pid = data["summary"]["participant_id"]
    for r in data["responses"]:
        r["condition"] = get_condition(r)
        r["participant_id"] = pid
        all_responses.append(r)
    participants.append({
        "id":       pid,
        "summary":  data["summary"],
        "emoji_freq": data["responses"][0]["demographics"].get("emojiFrequency", "Unknown"),
        "age":      data["responses"][0]["demographics"].get("age", None),
    })

n_participants = len(participants)
print(f"Loaded {n_participants} participant(s) from {len(files)} file(s).")

# ── Helper functions ──────────────────────────────────────────────────────────

def by_cond(cond):
    return [r for r in all_responses if r["condition"] == cond]

def acc(lst):
    s = [r for r in lst if r["is_correct"] is not None]
    return (sum(r["is_correct"] for r in s) / len(s) * 100) if s else 0

def control_accuracy(lst):
    scored = []

    for r in lst:
        q = r.get("question_text")
        correct = CONTROL_ANSWERS.get(q)

        if correct is not None:
            scored.append(r["response"] == correct)

    if not scored:
        return 0

    return (sum(scored) / len(scored)) * 100

def mean_rt(lst):
    return np.mean([r["reaction_time_seconds"] for r in lst]) if lst else 0

def mean_conf(lst):
    return np.mean([r["confidence"] for r in lst]) if lst else 0

conds      = ["Congruent", "Contradictory", "Control"]
cond_lists = {c: by_cond(c) for c in conds}

# Per-participant accuracy by condition
def per_participant_acc(cond):
    result = []

    for p in participants:
        pid = p["id"]
        lst = [r for r in cond_lists[cond] if r["participant_id"] == pid]

        scored = []

        for r in lst:
            if cond == "Control":
                q = r.get("question_text")
                correct = CONTROL_ANSWERS.get(q)
                if correct is not None:
                    scored.append(r["response"] == correct)
            else:
                if r.get("correct_answer") is not None:
                    scored.append(r["response"] == r["correct_answer"])

        if not scored:
            result.append(0)
        else:
            result.append((sum(scored) / len(scored)) * 100)

    return result

# Direction of errors in contradictory condition
contra_errors = [r for r in cond_lists["Contradictory"] if r.get("is_correct") is False]
emoji_biased  = sum(1 for r in contra_errors if r["response"] == r.get("correct_answer", "__") or
                    # followed emoji (wrong answer) not text
                    r["response"] != r.get("correct_answer"))
# More precisely: error = followed emoji sentiment (not text)
# correct_answer = text sentiment; error means response != correct_answer
emoji_direction_pct = (len(contra_errors) / max(len(cond_lists["Contradictory"]), 1)) * 100

# ── Print summary table ───────────────────────────────────────────────────────

print("\n── Aggregate Statistics ──────────────────────────────────────────────")
print(f"{'Condition':<16} {'Accuracy':>10} {'Mean RT (s)':>12} {'Mean Conf':>10} {'N responses':>12}")
print("-" * 62)
for c in conds:
    lst = cond_lists[c]
    print(f"{c:<16} {acc(lst):>9.1f}% {mean_rt(lst):>12.2f} {mean_conf(lst):>10.2f} {len(lst):>12}")

print(f"\nTotal participants : {n_participants}")
print(f"Total responses    : {len(all_responses)}")
print(f"Contradictory errors following emoji: {len(contra_errors)} / {len(cond_lists['Contradictory'])} "
      f"({emoji_direction_pct:.1f}%)")

# One-way repeated-measures style: per-participant means, then t-tests
cong_accs   = per_participant_acc("Congruent")
contra_accs = per_participant_acc("Contradictory")
ctrl_accs   = per_participant_acc("Control")

if n_participants >= 3:
    t1, p1 = stats.ttest_rel(cong_accs,   contra_accs)
    t2, p2 = stats.ttest_rel(ctrl_accs,   contra_accs)
    t3, p3 = stats.ttest_rel(cong_accs,   ctrl_accs)
    print(f"\nPaired t-tests (per-participant accuracy):")
    print(f"  Congruent vs Contradictory : t={t1:.2f}, p={p1:.4f}")
    print(f"  Control   vs Contradictory : t={t2:.2f}, p={p2:.4f}")
    print(f"  Congruent vs Control       : t={t3:.2f}, p={p3:.4f}")

# ── Styling ───────────────────────────────────────────────────────────────────

GOLD  = "#FFC72C"
DARK  = "#1a1a1a"
MID   = "#2a2a2a"
LIGHT = "#3a3a3a"
WHITE = "#f0f0f0"
RED   = "#e05252"
GREEN = "#52c07a"
BLUE  = "#5289e0"
COND_COLORS = {"Congruent": GREEN, "Contradictory": RED, "Control": BLUE}

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": MID,
    "axes.edgecolor": "#555", "axes.labelcolor": WHITE,
    "xtick.color": WHITE, "ytick.color": WHITE,
    "text.color": WHITE, "grid.color": "#444",
    "grid.linestyle": "--", "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
})

fig = plt.figure(figsize=(18, 14))
fig.suptitle(f"Emoji Sentiment Study  —  Aggregate Results  (N={n_participants})",
             fontsize=18, fontweight="bold", color=GOLD, y=0.98)

gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.38,
                      left=0.07, right=0.97, top=0.93, bottom=0.06)

# ── Row 0: summary cards ──────────────────────────────────────────────────────
ax_cards = fig.add_subplot(gs[0, :])
ax_cards.set_xlim(0, 1); ax_cards.set_ylim(0, 1); ax_cards.axis("off")

overall_acc   = acc([r for r in all_responses if r["is_correct"] is not None])
overall_rt    = mean_rt(all_responses)
overall_conf  = mean_conf(all_responses)
total_correct = sum(1 for r in all_responses if r.get("is_correct") is True)
total_wrong   = sum(1 for r in all_responses if r.get("is_correct") is False)

cards = [
    ("Participants",    str(n_participants),                    GOLD),
    ("Overall Accuracy", f"{overall_acc:.1f}%",                GREEN if overall_acc >= 60 else RED),
    ("Correct",         str(total_correct),                    GREEN),
    ("Incorrect",       str(total_wrong),                      RED),
    ("Mean RT",         f"{overall_rt:.2f}s",                  BLUE),
    ("Mean Confidence", f"{overall_conf:.2f}",                 "#aaa"),
]

card_w, card_h, pad = 0.145, 0.72, 0.012
for i, (label, value, color) in enumerate(cards):
    x = i * (card_w + pad) + 0.015
    rect = mpatches.FancyBboxPatch((x, 0.08), card_w, card_h,
                                   boxstyle="round,pad=0.02",
                                   facecolor=LIGHT, edgecolor=color, linewidth=1.5,
                                   transform=ax_cards.transAxes, clip_on=False)
    ax_cards.add_patch(rect)
    ax_cards.text(x + card_w/2, 0.62, value, ha="center", va="center",
                  fontsize=17, fontweight="bold", color=color,
                  transform=ax_cards.transAxes)
    ax_cards.text(x + card_w/2, 0.24, label, ha="center", va="center",
                  fontsize=9, color="#aaa", transform=ax_cards.transAxes)

# ── Row 1, col 0: Accuracy by condition (group means + individual dots) ───────
ax1 = fig.add_subplot(gs[1, 0])
accs = [
    acc(cond_lists[c]) if c != "Control" else control_accuracy(cond_lists[c])
    for c in conds
]
colors = [COND_COLORS[c] for c in conds]
bars   = ax1.bar(conds, accs, color=colors, edgecolor="#555", linewidth=0.8, width=0.5, zorder=2)
for bar, v in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 2, f"{v:.1f}%",
             ha="center", fontsize=10, fontweight="bold", color=WHITE)

# overlay individual participant dots
for ci, c in enumerate(conds):
    p_accs = per_participant_acc(c)
    jitter = np.random.uniform(-0.1, 0.1, len(p_accs))
    ax1.scatter([ci + j for j in jitter], p_accs,
                color=WHITE, s=30, zorder=5, alpha=0.7, edgecolors="#333", linewidths=0.5)

ax1.set_ylim(0, 115)
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Accuracy by Condition", fontweight="bold", color=GOLD, pad=8)
ax1.axhline(50, color="#888", linestyle="--", linewidth=1, label="Chance (50%)")
ax1.legend(fontsize=8, facecolor=MID, edgecolor="#555", labelcolor=WHITE)
ax1.grid(axis="y")

# ── Row 1, col 1: Reaction time by condition ──────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
rts    = [mean_rt(cond_lists[c]) for c in conds]
bars2  = ax2.bar(conds, rts, color=colors, edgecolor="#555", linewidth=0.8, width=0.5)
for bar, v in zip(bars2, rts):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.1, f"{v:.2f}s",
             ha="center", fontsize=10, fontweight="bold", color=WHITE)
ax2.set_ylabel("Mean RT (seconds)")
ax2.set_title("Reaction Time by Condition", fontweight="bold", color=GOLD, pad=8)
ax2.grid(axis="y")

# ── Row 1, col 2: Confidence by condition ─────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
confs  = [mean_conf(cond_lists[c]) for c in conds]
bars3  = ax3.bar(conds, confs, color=colors, edgecolor="#555", linewidth=0.8, width=0.5)
for bar, v in zip(bars3, confs):
    ax3.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}",
             ha="center", fontsize=10, fontweight="bold", color=WHITE)
ax3.set_ylim(0, 5.5)
ax3.set_ylabel("Mean Confidence (1–5)")
ax3.set_title("Confidence by Condition", fontweight="bold", color=GOLD, pad=8)
ax3.grid(axis="y")

# ── Row 2, col 0-1: Per-participant accuracy grouped bar ─────────────────────
ax4 = fig.add_subplot(gs[2, :2])
pids     = [p["id"] for p in participants]
x        = np.arange(len(pids))
width    = 0.25
cong_a   = per_participant_acc("Congruent")
contra_a = per_participant_acc("Contradictory")
ctrl_a   = per_participant_acc("Control")

ax4.bar(x - width, cong_a,   width, label="Congruent",     color=GREEN, edgecolor="#333")
ax4.bar(x,         contra_a, width, label="Contradictory", color=RED,   edgecolor="#333")
ax4.bar(x + width, ctrl_a,   width, label="Control",       color=BLUE,  edgecolor="#333")
ax4.set_xticks(x)
ax4.set_xticklabels(pids, fontsize=9, rotation=30, ha="right")
ax4.set_ylabel("Accuracy (%)")
ax4.set_ylim(0, 115)
ax4.set_title("Per-Participant Accuracy by Condition", fontweight="bold", color=GOLD, pad=8)
ax4.axhline(50, color="#888", linestyle="--", linewidth=1)
ax4.legend(fontsize=8, facecolor=MID, edgecolor="#555", labelcolor=WHITE)
ax4.grid(axis="y")

# ── Row 2, col 2: Emoji freq vs contradictory accuracy scatter ───────────────
ax5 = fig.add_subplot(gs[2, 2])
freq_map = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Always": 5}
xs, ys, labels = [], [], []
for p, ca in zip(participants, contra_a):
    fv = freq_map.get(p["emoji_freq"])
    if fv is not None:
        xs.append(fv); ys.append(ca); labels.append(p["id"])

ax5.scatter(xs, ys, color=RED, s=80, edgecolors="#333", linewidths=0.8, zorder=3)
for xi, yi, lab in zip(xs, ys, labels):
    ax5.annotate(lab, (xi, yi), textcoords="offset points", xytext=(5, 3),
                 fontsize=8, color="#ccc")

if len(xs) >= 3:
    m, b, r, p_val, _ = stats.linregress(xs, ys)
    xfit = np.linspace(min(xs), max(xs), 100)
    ax5.plot(xfit, m*xfit + b, color=GOLD, linewidth=1.5, linestyle="--",
             label=f"r={r:.2f}, p={p_val:.3f}")
    ax5.legend(fontsize=8, facecolor=MID, edgecolor="#555", labelcolor=WHITE)

ax5.set_xticks([1,2,3,4,5])
ax5.set_xticklabels(["Never","Rarely","Some","Often","Always"], fontsize=8)
ax5.set_ylabel("Contradictory Accuracy (%)")
ax5.set_ylim(0, 105)
ax5.set_title("Emoji Use vs Contradictory Accuracy", fontweight="bold", color=GOLD, pad=8)
ax5.grid()

out_path = "aggregate_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"Saved chart → {out_path}")

# ── Demographics chart ────────────────────────────────────────────────────────

# Pull one demographics record per participant
demos = [data["responses"][0]["demographics"]
         for path in files
         for data in [json.load(open(path, encoding="utf-8"))]]

ages             = [d.get("age") for d in demos if d.get("age")]
freq_counts      = {}
style_counts     = {}
platform_counts  = {}
interp_scores    = [d.get("interpretConfidence") for d in demos if d.get("interpretConfidence")]
sarcasm_scores   = [d.get("sarcasmDetection")    for d in demos if d.get("sarcasmDetection")]

FREQ_ORDER  = ["Never", "Rarely", "Sometimes", "Often", "Always"]
STYLE_ORDER = ["Very casual / slang-heavy", "Casual but clear",
               "Neutral / depends on context", "Formal and complete sentences"]

for d in demos:
    f = d.get("emojiFrequency")
    if f: freq_counts[f] = freq_counts.get(f, 0) + 1
    s = d.get("textingStyle")
    if s: style_counts[s] = style_counts.get(s, 0) + 1
    for p in d.get("platforms", []):
        platform_counts[p] = platform_counts.get(p, 0) + 1

fig2 = plt.figure(figsize=(18, 12))
fig2.suptitle(f"Participant Demographics  (N={n_participants})",
              fontsize=18, fontweight="bold", color=GOLD, y=0.98)

gs2 = fig2.add_gridspec(2, 3, hspace=0.55, wspace=0.4,
                         left=0.07, right=0.97, top=0.92, bottom=0.08)

# Age distribution
ax_age = fig2.add_subplot(gs2[0, 0])
if ages:
    ax_age.hist(ages, bins=range(min(ages), max(ages)+2), color=GOLD,
                edgecolor="#333", linewidth=0.8)
ax_age.set_xlabel("Age")
ax_age.set_ylabel("Count")
ax_age.set_title("Age Distribution", fontweight="bold", color=GOLD, pad=8)
ax_age.grid(axis="y")

# Emoji frequency
ax_freq = fig2.add_subplot(gs2[0, 1])
freq_labels = [f for f in FREQ_ORDER if f in freq_counts]
freq_vals   = [freq_counts[f] for f in freq_labels]
bars_f = ax_freq.bar(freq_labels, freq_vals, color=BLUE, edgecolor="#333", linewidth=0.8)
for bar, v in zip(bars_f, freq_vals):
    ax_freq.text(bar.get_x() + bar.get_width()/2, v + 0.1, str(v),
                 ha="center", fontsize=11, fontweight="bold", color=WHITE)
ax_freq.set_ylabel("Count")
ax_freq.set_title("Emoji Use Frequency", fontweight="bold", color=GOLD, pad=8)
ax_freq.set_ylim(0, max(freq_vals or [1]) + 2)
ax_freq.tick_params(axis="x", labelsize=9)
ax_freq.grid(axis="y")

# Texting style
ax_style = fig2.add_subplot(gs2[0, 2])
style_labels = [s for s in STYLE_ORDER if s in style_counts]
style_vals   = [style_counts[s] for s in style_labels]
short_labels = ["Very casual", "Casual/clear", "Neutral", "Formal"][:len(style_labels)]
bars_s = ax_style.bar(short_labels, style_vals, color=GREEN, edgecolor="#333", linewidth=0.8)
for bar, v in zip(bars_s, style_vals):
    ax_style.text(bar.get_x() + bar.get_width()/2, v + 0.1, str(v),
                  ha="center", fontsize=11, fontweight="bold", color=WHITE)
ax_style.set_ylabel("Count")
ax_style.set_title("Texting Style", fontweight="bold", color=GOLD, pad=8)
ax_style.set_ylim(0, max(style_vals or [1]) + 2)
ax_style.grid(axis="y")

# Platforms
ax_plat = fig2.add_subplot(gs2[1, 0])
plat_labels = sorted(platform_counts, key=platform_counts.get, reverse=True)
plat_vals   = [platform_counts[p] for p in plat_labels]
short_plat  = [p.replace(" / ", "/").replace("Twitter/X DMs", "X DMs") for p in plat_labels]
bars_p = ax_plat.barh(short_plat[::-1], plat_vals[::-1], color=BLUE, edgecolor="#333", linewidth=0.8)
for bar, v in zip(bars_p, plat_vals[::-1]):
    ax_plat.text(v + 0.05, bar.get_y() + bar.get_height()/2, str(v),
                 va="center", fontsize=10, fontweight="bold", color=WHITE)
ax_plat.set_xlabel("Count")
ax_plat.set_title("Platforms Used", fontweight="bold", color=GOLD, pad=8)
ax_plat.set_xlim(0, max(plat_vals or [1]) + 1.5)
ax_plat.grid(axis="x")

# Emoji interpretation confidence (1-7)
ax_interp = fig2.add_subplot(gs2[1, 1])
interp_counts = [interp_scores.count(i) for i in range(1, 8)]
bars_i = ax_interp.bar(range(1, 8), interp_counts, color=GREEN, edgecolor="#333", linewidth=0.8)
for bar, v in zip(bars_i, interp_counts):
    if v > 0:
        ax_interp.text(bar.get_x() + bar.get_width()/2, v + 0.05, str(v),
                       ha="center", fontsize=10, fontweight="bold", color=WHITE)
ax_interp.set_xlabel("Score (1 = not at all, 7 = extremely)")
ax_interp.set_ylabel("Count")
ax_interp.set_title("Emoji Interpretation Confidence", fontweight="bold", color=GOLD, pad=8)
ax_interp.set_xticks(range(1, 8))
if interp_scores:
    ax_interp.axvline(np.mean(interp_scores), color=GOLD, linestyle="--", linewidth=1.5,
                      label=f"Mean: {np.mean(interp_scores):.1f}")
    ax_interp.legend(fontsize=8, facecolor=MID, edgecolor="#555", labelcolor=WHITE)
ax_interp.grid(axis="y")

# Sarcasm detection (1-7)
ax_sarc = fig2.add_subplot(gs2[1, 2])
sarc_counts = [sarcasm_scores.count(i) for i in range(1, 8)]
bars_sa = ax_sarc.bar(range(1, 8), sarc_counts, color=RED, edgecolor="#333", linewidth=0.8)
for bar, v in zip(bars_sa, sarc_counts):
    if v > 0:
        ax_sarc.text(bar.get_x() + bar.get_width()/2, v + 0.05, str(v),
                     ha="center", fontsize=10, fontweight="bold", color=WHITE)
ax_sarc.set_xlabel("Score (1 = not at all, 7 = extremely)")
ax_sarc.set_ylabel("Count")
ax_sarc.set_title("Sarcasm Detection Confidence", fontweight="bold", color=GOLD, pad=8)
ax_sarc.set_xticks(range(1, 8))
if sarcasm_scores:
    ax_sarc.axvline(np.mean(sarcasm_scores), color=GOLD, linestyle="--", linewidth=1.5,
                    label=f"Mean: {np.mean(sarcasm_scores):.1f}")
    ax_sarc.legend(fontsize=8, facecolor=MID, edgecolor="#555", labelcolor=WHITE)
ax_sarc.grid(axis="y")

demo_path = "aggregate_demographics.png"
plt.savefig(demo_path, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"Saved chart → {demo_path}")