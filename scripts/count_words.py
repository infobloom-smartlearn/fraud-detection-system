import re
from pathlib import Path

text = Path("docs/chapters-4-8.md").read_text(encoding="utf-8")

targets = {
    "4.1": 200, "4.2": 250, "4.3": 250, "4.4": 200, "4.5": 400, "4.6": 200, "4.7": 100,
    "5.1": 100, "5.2": 120, "5.3": 100, "5.4": 100, "5.5": 80,
    "6.1": 80, "6.2": 220, "6.3": 250, "6.4": 180, "6.5": 100, "6.6": 120, "6.7": 50,
    "7.1": 120, "7.2": 100, "7.3": 60, "7.4": 70,
    "8.1": 80, "8.2": 120, "8.3": 50, "8.4": 100,
}


def wc(body: str) -> int:
    body = re.sub(r"```[\s\S]*?```", " ", body)
    body = re.sub(r"\*\*([^*]+)\*\*", r"\1", body)
    body = re.sub(r"\*([^*]+)\*", r"\1", body)
    body = re.sub(r"\[[^\]]+\]", " ", body)
    body = re.sub(r"`([^`]+)`", r"\1", body)
    body = re.sub(r"\|[^|\n]+\|", " ", body)
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", body)
    return len(words)


parts = re.split(r"\n(?=## \d+\.\d+)", text)
chapter_totals = {4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
for p in parts:
    if not p.strip().startswith("##"):
        continue
    num = re.match(r"## (\d\.\d+)", p).group(1)
    body = p.split("\n", 1)[1] if "\n" in p else ""
    w = wc(body)
    ch = int(num.split(".")[0])
    chapter_totals[ch] += w
    t = targets.get(num, 0)
    print(f"  {num:4} target={t:4} actual={w:4} off={w-t:+d}")

print("---")
for ch, total in [(4, 1600), (5, 500), (6, 1000), (7, 350), (8, 350)]:
    print(f"Chapter {ch}: {chapter_totals[ch]} / {total} (off {chapter_totals[ch]-total:+d})")
