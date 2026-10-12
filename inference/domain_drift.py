"""How much do continuations of GENERAL prompts drift into After-Sales vocabulary?

Usage (from the project root):
    python -m inference.domain_drift --checkpoint checkpoints/transformer_bpe4096/best.pt

For every general prompt in configs/eval_prompts.json, several continuations are sampled
(one seed each). We count After-Sales terms per 1,000 characters and compare with the same
rate in real held-out text: general text (what a non-drifting model should look like) and
After-Sales text (full drift).
"""

import argparse
import json
import re
from pathlib import Path

import torch

from inference.generate import load_checkpoint
from inference.sample_eval import generate_chars
from training.config import TrainConfig
from training.train import ROOT

# Words that are typical for the After-Sales corpus and rare in general text.
DOMAIN_TERMS = [
    "التوصيل", "التركيب", "الصيانة", "التصنيع", "المقاسات", "القياسات", "الفاتورة", "فاتورة", "الفواتير",
    "المستودع", "الفني", "المهمة", "التذكرة", "الشكوى", "خدمة العملاء", "ما بعد البيع",
    "Odoo", "SAP", "invoice", "installation", "delivery", "warehouse", "technician", "ticket", "appointment",
]
_TERMS = re.compile("|".join(map(re.escape, DOMAIN_TERMS)), re.IGNORECASE)


def domain_rate(text):
    """After-Sales terms per 1,000 characters."""
    return 1000 * len(_TERMS.findall(text)) / max(len(text), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", default="configs/eval_prompts.json")
    parser.add_argument("--samples", type=int, default=10, help="continuations per prompt")
    parser.add_argument("--chars", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    model, tokenizer, ckpt = load_checkpoint(args.checkpoint, "cpu")
    cfg = TrainConfig.from_dict(ckpt["config"])
    dataset = ROOT / cfg.dataset
    reference = {
        "general_ar held-out": "\n".join((dataset / "val" / f"{n}.txt").read_text(encoding="utf-8")
                                        for n in ("common_voice_ar", "ntrex_ar", "xquad_ar")),
        "general_en held-out": "\n".join((dataset / "val" / f"{n}.txt").read_text(encoding="utf-8")
                                        for n in ("common_voice_en", "gutenberg_en")),
        "after_sales held-out": (dataset / "val" / "after_sales.txt").read_text(encoding="utf-8"),
    }
    report = {"checkpoint": args.checkpoint, "settings": vars(args),
              "reference_rates": {k: round(domain_rate(v), 2) for k, v in reference.items()}, "groups": {}}

    prompts = json.loads((ROOT / args.prompts).read_text(encoding="utf-8"))
    seed = args.seed
    for group in ("general_ar", "general_en"):
        texts = []
        for prompt in prompts[group]:
            for _ in range(args.samples):
                gen = torch.Generator().manual_seed(seed)
                seed += 1
                texts.append(generate_chars(model, tokenizer, prompt, args.chars, args.temperature, args.top_k, gen))
        joined = "\n".join(texts)
        report["groups"][group] = {
            "samples": len(texts),
            "domain_terms_per_1000_chars": round(domain_rate(joined), 2),
            "samples_with_any_domain_term": sum(bool(_TERMS.search(t)) for t in texts),
        }

    print(json.dumps({k: report[k] for k in ("reference_rates", "groups")}, ensure_ascii=False, indent=2))
    out = Path(args.checkpoint).with_name(f"drift_{Path(args.checkpoint).stem}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
