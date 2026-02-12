#!/usr/bin/env python3
"""
Stage_2_Instruction_Tuning_script_v11_global.py

Purpose
-------
A drop-in Stage-2 instruction tuning script (Phi-4 / causal LM) with:
- tune_strategy: lm_head | last_n | all (train more than heads via last N transformer blocks)
- target-only SFT by default (prompt masked) + optional label_prompt_last_n ablation
- Global instruction templates injected in-code (single template) OR template pools (Gemini/ChatGPT variants)
- Optional prompt "styles": original / minimal / chat
- Optional prompt/target canonicalization switches (for overlap metrics and SER evaluation consistency)

This script is designed to match the CLI interface you have been using:
  --train_file, --val_file, --base_model_dir, --output_dir, --tune_strategy, --unfreeze_last_n, ...

Notes
-----
- Default behavior: target-only loss (prompt tokens masked to -100)
- By default, instructions come from the dataset's prompt field. You can override with:
    --global_instruction_file ... and --global_instruction_mode ...
  or
    --instruction_variants_csv ... (Gemini/ChatGPT pools)

Author: ChatGPT (v11)
"""

import argparse
import csv
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

try:
    import psutil
except Exception:
    psutil = None


# ----------------------------
# Global instruction defaults
# ----------------------------

GLOBAL_INSTRUCTION_STRICT = """You are an air traffic controller. Produce exactly ONE ATC radio transmission.

Hard constraints:
- Output only the spoken ATC line (single line). No headers, no bullets, no explanations.
- Begin with the aircraft callsign exactly as provided.
- Use standard FAA/ICAO phraseology.
- Copy all required slot values exactly from the input when present (runway, taxiways, hold short, altitude, heading, speed, frequency).
- Do NOT invent new instructions, restrictions, numbers, or weather not present in the input.
- Keep it concise and operational.
"""


# ----------------------------
# Helpers: memory + logging
# ----------------------------

def print_mem(prefix: str) -> None:
    print(f"\n🧠 Memory Status {prefix}:")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"GPU Memory Allocated: {alloc:.2f} GB")
        print(f"GPU Memory Reserved:  {reserved:.2f} GB")
    else:
        print("GPU not available.")

    if psutil is not None:
        cpu = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        print(f"CPU Memory Used:      {cpu:.2f} GB\n")
    else:
        print("CPU Memory Used:      (psutil not installed)\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Prompt parsing + injection
# ----------------------------

_SECTION_RE = re.compile(
    r"(?is)"
    r"(?:^\s*\[Instruction\]\s*:\s*(?P<instr>.*?))?"
    r"(?:^\s*\[Scenario\]\s*:\s*(?P<scen>.*?))?"
    r"(?:^\s*\[Pilot\]\s*:\s*(?P<pilot>.*))?$"
)

def split_prompt_sections(prompt: str) -> Tuple[str, str, str]:
    """
    Attempts to parse prompts of the form:
      [Instruction]: ...
      [Scenario]: ...
      [Pilot]: ...
    Falls back gracefully if missing sections.
    """
    p = (prompt or "").strip()
    instr, scen, pilot = "", "", ""
    if "[Instruction]" in p or "[Scenario]" in p or "[Pilot]" in p:
        # Robust line-based split
        current = None
        buf = {"instr": [], "scen": [], "pilot": []}
        for line in p.splitlines():
            l = line.strip()
            if l.lower().startswith("[instruction]"):
                current = "instr"
                buf[current].append(line.split(":", 1)[1].strip() if ":" in line else "")
                continue
            if l.lower().startswith("[scenario]"):
                current = "scen"
                buf[current].append(line.split(":", 1)[1].strip() if ":" in line else "")
                continue
            if l.lower().startswith("[pilot]"):
                current = "pilot"
                buf[current].append(line.split(":", 1)[1].strip() if ":" in line else "")
                continue
            if current is None:
                # no header yet, treat as scenario preamble
                buf["scen"].append(line)
            else:
                buf[current].append(line)
        instr = "\n".join(buf["instr"]).strip()
        scen = "\n".join(buf["scen"]).strip()
        pilot = "\n".join(buf["pilot"]).strip()
        return instr, scen, pilot

    # No tags found; treat whole thing as scenario
    return "", p, ""


def normalize_ws(s: str) -> str:
    return re.sub(r"[ \t]+", " ", (s or "").strip())


def canonicalize_text(s: str, remove_punct: bool = False) -> str:
    s = normalize_ws(s)
    if remove_punct:
        s = re.sub(r"[^\w\s\[\]\:\-\/\.]", "", s)
    return s


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_instruction_variants_csv(path: str) -> List[Dict[str, str]]:
    """
    Expects columns similar to your Gemini CSV:
      Template Name, Full Instruction Content, Script Implementation, Benefit
    Returns list of dict rows.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def choose_variant(variants: List[Dict[str, str]], policy: str, rng: random.Random) -> Optional[str]:
    if not variants:
        return None
    if policy == "first":
        return (variants[0].get("Full Instruction Content") or "").strip()
    if policy == "random":
        return (rng.choice(variants).get("Full Instruction Content") or "").strip()
    if policy == "cycle":
        # We'll cycle deterministically by RNG state; caller should keep rng seeded
        idx = rng.randint(0, 10**9) % len(variants)
        return (variants[idx].get("Full Instruction Content") or "").strip()
    return (variants[0].get("Full Instruction Content") or "").strip()


def build_prompt(
    dataset_prompt: str,
    prompt_style: str,
    global_instruction: Optional[str],
    global_mode: str,
    variants: Optional[List[Dict[str, str]]],
    variant_policy: str,
    canonicalize_prompts: bool,
    canonical_remove_punct: bool,
    move_atc_prefix_to_prompt: bool,
    rng: random.Random,
) -> str:
    """
    Returns the final prompt_text to feed the tokenizer.
    """
    orig_instr, scen, pilot = split_prompt_sections(dataset_prompt)
    orig_instr = orig_instr.strip()

    # Optionally move a leading "[ATC]:" or "ATC:" prefix out of target into prompt (rarely needed here).
    if move_atc_prefix_to_prompt:
        # This is a placeholder hook; actual behavior depends on your dataset format.
        # We leave it as prompt-side context.
        pass

    # Decide instruction text
    instr_text = orig_instr

    # If variants provided, they take precedence when global_mode is "variants"
    if variants:
        picked = choose_variant(variants, variant_policy, rng)
        if picked:
            instr_text = picked

    # Global instruction injection
    if global_instruction and global_mode != "none":
        gi = global_instruction.strip()
        if global_mode == "replace_original":
            instr_text = gi
        elif global_mode == "wrap_original":
            if orig_instr:
                instr_text = gi + "\nTask: " + orig_instr
            else:
                instr_text = gi
        elif global_mode == "prepend_original":
            if orig_instr:
                instr_text = gi + "\n" + orig_instr
            else:
                instr_text = gi
        elif global_mode == "minimal_only":
            instr_text = gi

    # Canonicalize prompt fields if requested
    if canonicalize_prompts:
        instr_text = canonicalize_text(instr_text, remove_punct=canonical_remove_punct)
        scen = canonicalize_text(scen, remove_punct=canonical_remove_punct)
        pilot = canonicalize_text(pilot, remove_punct=canonical_remove_punct)

    # Compose by style
    if prompt_style == "original":
        parts = []
        if instr_text:
            parts.append(f"[Instruction]: {instr_text}")
        if scen:
            parts.append(f"[Scenario]: {scen}")
        if pilot:
            parts.append(f"[Pilot]: {pilot}")
        return "\n".join(parts).strip()

    if prompt_style == "minimal":
        # Minimal ignores dataset instruction unless global_mode requests wrap/prepend
        if global_instruction and global_mode in ("replace_original", "wrap_original", "prepend_original", "minimal_only"):
            short_sys = instr_text.strip()
        else:
            short_sys = "You are ATC. Respond with ONE ATC radio line only. No explanations."
        parts = [f"[Instruction]: {short_sys}"]
        if scen:
            parts.append(f"[Scenario]: {scen}")
        if pilot:
            parts.append(f"[Pilot]: {pilot}")
        return "\n".join(parts).strip()

    if prompt_style == "chat":
        # Chat-like wrapper that helps some models separate instruction from user content
        sys = instr_text.strip() if instr_text else "You are ATC."
        user_parts = []
        if scen:
            user_parts.append(f"[Scenario]: {scen}")
        if pilot:
            user_parts.append(f"[Pilot]: {pilot}")
        user = "\n".join(user_parts).strip()
        return f"### System:\n{sys}\n\n### User:\n{user}\n\n### Assistant:\n"

    raise ValueError(f"Unknown prompt_style={prompt_style}")


# ----------------------------
# Dataset
# ----------------------------

def infer_columns(df_columns: List[str]) -> Tuple[str, str]:
    cols = [c.strip() for c in df_columns]
    # Prompt candidates
    prompt_cands = ["prompt", "input", "instruction", "source", "question"]
    target_cands = ["target", "output", "response", "completion", "answer"]
    prompt_col = None
    target_col = None
    lower_map = {c.lower(): c for c in cols}
    for c in prompt_cands:
        if c in lower_map:
            prompt_col = lower_map[c]; break
    for c in target_cands:
        if c in lower_map:
            target_col = lower_map[c]; break
    if prompt_col is None or target_col is None:
        raise ValueError(f"Could not infer prompt/target columns. Found columns={cols}. "
                         f"Expected one of {prompt_cands} and one of {target_cands}.")
    return prompt_col, target_col


class Stage2Dataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer: AutoTokenizer,
        max_length: int,
        prompt_style: str,
        global_instruction: Optional[str],
        global_mode: str,
        variants: Optional[List[Dict[str, str]]],
        variant_policy: str,
        canonicalize_prompts: bool,
        canonicalize_targets: bool,
        canonical_remove_punct: bool,
        move_atc_prefix_to_prompt: bool,
        label_prompt_last_n: int,
        seed: int,
    ):
        import pandas as pd

        self.df = pd.read_csv(csv_path)
        self.prompt_col, self.target_col = infer_columns(list(self.df.columns))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_style = prompt_style
        self.global_instruction = global_instruction
        self.global_mode = global_mode
        self.variants = variants
        self.variant_policy = variant_policy
        self.canonicalize_prompts = canonicalize_prompts
        self.canonicalize_targets = canonicalize_targets
        self.canonical_remove_punct = canonical_remove_punct
        self.move_atc_prefix_to_prompt = move_atc_prefix_to_prompt
        self.label_prompt_last_n = max(0, int(label_prompt_last_n))
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        raw_prompt = str(row[self.prompt_col])
        raw_target = str(row[self.target_col])

        prompt_text = build_prompt(
            raw_prompt,
            prompt_style=self.prompt_style,
            global_instruction=self.global_instruction,
            global_mode=self.global_mode,
            variants=self.variants,
            variant_policy=self.variant_policy,
            canonicalize_prompts=self.canonicalize_prompts,
            canonical_remove_punct=self.canonical_remove_punct,
            move_atc_prefix_to_prompt=self.move_atc_prefix_to_prompt,
            rng=self.rng,
        ).strip()

        target_text = raw_target.strip()

        if self.canonicalize_targets:
            target_text = canonicalize_text(target_text, remove_punct=self.canonical_remove_punct)

        # Tokenize separately
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        target_ids = self.tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            # fall back to sep or pad if needed
            eos_id = self.tokenizer.sep_token_id or self.tokenizer.pad_token_id

        # Build input_ids
        input_ids = prompt_ids + target_ids + ([eos_id] if eos_id is not None else [])
        input_ids = input_ids[: self.max_length]

        # Build labels
        # Default: prompt masked; target supervised
        labels = [-100] * len(prompt_ids) + target_ids + ([eos_id] if eos_id is not None else [])
        labels = labels[: self.max_length]

        # Optional ablation: label last N prompt tokens too
        if self.label_prompt_last_n > 0 and len(prompt_ids) > 0:
            n = min(self.label_prompt_last_n, len(prompt_ids))
            for j in range(len(prompt_ids) - n, len(prompt_ids)):
                if j < len(labels):
                    labels[j] = input_ids[j]

        attn = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_text": prompt_text,
            "target_text": target_text,
        }


@dataclass
class Collator:
    tokenizer: AutoTokenizer

    def __call__(self, batch):
        # Pad to max length in batch
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ----------------------------
# Trainable params control
# ----------------------------

def freeze_all(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def set_trainable_lm_head(model: torch.nn.Module):
    for n, p in model.named_parameters():
        if "lm_head" in n:
            p.requires_grad = True


def set_trainable_last_n(model: torch.nn.Module, n_last: int):
    """
    Unfreeze lm_head + last n transformer blocks if model has a standard HF layout.
    Works for many decoder-only LMs (LLaMA-like).
    """
    set_trainable_lm_head(model)

    # Try common attribute paths
    blocks = None
    for path in [
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("model", "transformer", "h"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]:
        cur = model
        ok = True
        for attr in path:
            if hasattr(cur, attr):
                cur = getattr(cur, attr)
            else:
                ok = False
                break
        if ok and isinstance(cur, (list, torch.nn.ModuleList)):
            blocks = cur
            break

    if blocks is None:
        print("[WARN] Could not locate transformer blocks automatically; falling back to lm_head only.")
        return

    n_last = max(0, int(n_last))
    if n_last == 0:
        return

    for blk in list(blocks)[-n_last:]:
        for p in blk.parameters():
            p.requires_grad = True


def count_trainable(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ----------------------------
# Validation
# ----------------------------

@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        out = model(**batch)
        loss = out.loss.detach().float().item()
        losses.append(loss)
    model.train()
    return float(sum(losses) / max(1, len(losses)))


# ----------------------------
# Main
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_file", required=True, type=str)
    ap.add_argument("--val_file", required=True, type=str)
    ap.add_argument("--base_model_dir", required=True, type=str)
    ap.add_argument("--output_dir", required=True, type=str)

    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_every_steps", type=int, default=250)

    ap.add_argument("--tune_strategy", type=str, default="lm_head", choices=["lm_head", "last_n", "all"])
    ap.add_argument("--unfreeze_last_n", type=int, default=4)

    ap.add_argument("--canonicalize_targets", action="store_true")
    ap.add_argument("--canonicalize_prompts", action="store_true")
    ap.add_argument("--canonical_remove_punct", action="store_true")

    ap.add_argument("--move_atc_prefix_to_prompt", action="store_true")
    ap.add_argument("--label_prompt_last_n", type=int, default=0)

    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)
    ap.add_argument("--gradient_checkpointing", action="store_true")

    ap.add_argument("--early_stop_patience", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=0.0)

    # Prompt + global instruction controls
    ap.add_argument("--prompt_style", type=str, default="original", choices=["original", "minimal", "chat"])

    ap.add_argument("--global_instruction_file", type=str, default=None,
                    help="Path to a txt file containing a global instruction template.")
    ap.add_argument("--global_instruction_mode", type=str, default="none",
                    choices=["none", "replace_original", "wrap_original", "prepend_original", "minimal_only"],
                    help="How to apply the global instruction template to dataset prompts.")

    ap.add_argument("--instruction_variants_csv", type=str, default=None,
                    help="CSV with instruction variants (e.g., Gemini/ChatGPT template pool).")
    ap.add_argument("--instruction_variant_policy", type=str, default="first",
                    choices=["first", "random", "cycle"],
                    help="How to choose a variant from instruction_variants_csv.")

    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("[INFO] Loading datasets...")
    import pandas as pd
    train_df = pd.read_csv(args.train_file)
    val_df = pd.read_csv(args.val_file)
    print(f"[INFO] train={len(train_df)} rows, val={len(val_df)} rows")

    print("[INFO] Loading tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(args.base_model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if args.gradient_checkpointing:
        print("[INFO] Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    # Trainability
    freeze_all(model)
    if args.tune_strategy == "lm_head":
        set_trainable_lm_head(model)
    elif args.tune_strategy == "last_n":
        set_trainable_last_n(model, args.unfreeze_last_n)
    elif args.tune_strategy == "all":
        for p in model.parameters():
            p.requires_grad = True

    trainable, total = count_trainable(model)
    pct = 100.0 * trainable / max(1, total)
    print(f"[INFO] tune_strategy={args.tune_strategy} | trainable={trainable:,} / total={total:,} ({pct:.4f}%)")
    print(f"[INFO] move_atc_prefix_to_prompt={args.move_atc_prefix_to_prompt}")
    print(f"[INFO] label_prompt_last_n={args.label_prompt_last_n}")

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Load global instruction and/or variants
    global_instruction = None
    if args.global_instruction_file:
        global_instruction = load_text_file(args.global_instruction_file)
        print(f"[INFO] Loaded global_instruction_file={args.global_instruction_file} ({len(global_instruction)} chars)")
    else:
        # default: None (use dataset instructions). You can uncomment to force a default template:
        # global_instruction = GLOBAL_INSTRUCTION_STRICT
        pass

    variants = None
    if args.instruction_variants_csv:
        variants = load_instruction_variants_csv(args.instruction_variants_csv)
        print(f"[INFO] Loaded instruction_variants_csv={args.instruction_variants_csv} rows={len(variants)} policy={args.instruction_variant_policy}")

    # Create datasets
    train_ds = Stage2Dataset(
        csv_path=args.train_file,
        tokenizer=tok,
        max_length=args.max_length,
        prompt_style=args.prompt_style,
        global_instruction=global_instruction,
        global_mode=args.global_instruction_mode,
        variants=variants,
        variant_policy=args.instruction_variant_policy,
        canonicalize_prompts=args.canonicalize_prompts,
        canonicalize_targets=args.canonicalize_targets,
        canonical_remove_punct=args.canonical_remove_punct,
        move_atc_prefix_to_prompt=args.move_atc_prefix_to_prompt,
        label_prompt_last_n=args.label_prompt_last_n,
        seed=args.seed,
    )

    val_ds = Stage2Dataset(
        csv_path=args.val_file,
        tokenizer=tok,
        max_length=args.max_length,
        prompt_style=args.prompt_style,
        global_instruction=global_instruction,
        global_mode=args.global_instruction_mode,
        variants=variants,
        variant_policy="first",
        canonicalize_prompts=args.canonicalize_prompts,
        canonicalize_targets=args.canonicalize_targets,
        canonical_remove_punct=args.canonical_remove_punct,
        move_atc_prefix_to_prompt=args.move_atc_prefix_to_prompt,
        label_prompt_last_n=args.label_prompt_last_n,
        seed=args.seed + 1,
    )

    collate = Collator(tok)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Optimizer/scheduler
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Device (pick first param device)
    device = next(model.parameters()).device

    print_mem("Pre-Training")
    print("\n🚀 Starting Stage-2 instruction tuning (V11)...\n")

    best_val = float("inf")
    bad_epochs = 0
    global_step = 0
    optimizer_step = 0
    t0 = time.time()

    model.train()

    for epoch in range(1, args.epochs + 1):
        print(f"[INFO] Epoch {epoch}/{args.epochs}")
        running = 0.0
        count = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            for k in batch:
                batch[k] = batch[k].to(device)

            out = model(**batch)
            loss = out.loss / max(1, args.grad_accum)
            loss.backward()

            running += loss.detach().float().item()
            count += 1
            global_step += 1

            if global_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                optimizer_step += 1

                if optimizer_step % 25 == 0:
                    lr = sched.get_last_lr()[0]
                    print(f"[TRAIN] step={optimizer_step} loss={running/max(1,count):.4f} lr={lr:.2e}")
                    running = 0.0
                    count = 0

                # Validation cadence
                if optimizer_step % args.val_every_steps == 0:
                    v = eval_loss(model, val_loader, device)
                    print(f"[VAL] step={optimizer_step} mean_loss={v:.4f}")
                    if v + args.min_delta < best_val:
                        best_val = v
                        bad_epochs = 0
                        print(f"[VAL] ✅ New best val loss={best_val:.4f} -> saving to {args.output_dir}")
                        model.save_pretrained(args.output_dir)
                        tok.save_pretrained(args.output_dir)
                    else:
                        bad_epochs += 1
                        print(f"[VAL] no improvement (best={best_val:.4f}) bad_epochs={bad_epochs}")
                        if bad_epochs >= args.early_stop_patience:
                            print("[INFO] Early stopping triggered.")
                            elapsed = time.time() - t0
                            print(f"\n✅ Done. Total optimizer steps={optimizer_step}. Elapsed={elapsed/60:.1f} min")
                            print_mem("Post-Training")
                            return

    elapsed = time.time() - t0
    print(f"\n✅ Done. Total optimizer steps={optimizer_step}. Elapsed={elapsed/60:.1f} min")
    print_mem("Post-Training")

    # Final save (best already saved, but keep a final snapshot too)
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    print(f"[INFO] Saving final snapshot to {final_dir}")
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
