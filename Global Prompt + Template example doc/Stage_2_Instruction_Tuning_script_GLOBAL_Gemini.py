#!/usr/bin/env python3
"""
Stage_2_Instruction_Tuning_script_v11.py
Updated with Global System Prompt support.
"""

import os
import gc
import time
import json
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

# ==========================================
# 1. DEFINE YOUR GLOBAL INSTRUCTION HERE
# ==========================================
GLOBAL_SYSTEM_PROMPT = (
    "You are an expert Air Traffic Controller. Provide exactly ONE ATC radio line. "
    "Start with the callsign. Preserve all runways, altitudes, and technical data exactly "
    "as provided. Do not add invented details or explanations."
)

class Stage2Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512, global_prompt=""):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = max_length
        self.global_prompt = global_prompt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 2. WHERE THE INSTRUCTION IS PROMPTED
        # We strip the old instructions from the CSV and prepend the global one
        raw_prompt = str(self.df.loc[idx, "prompt"])
        
        # Logic to clean existing instruction if present in CSV
        if "[Scenario]:" in raw_prompt:
            scenario_part = raw_prompt.split("[Scenario]:")[-1]
            clean_prompt = f"[Scenario]:{scenario_part}"
        else:
            clean_prompt = raw_prompt

        # PREPENDING THE GLOBAL INSTRUCTION
        full_input_text = f"{self.global_prompt}\n\n{clean_prompt}\n[ATC]:"
        target_text = " " + str(self.df.loc[idx, "target"]).replace("[ATC]:", "").strip()

        # Tokenization & Masking
        prompt_ids = self.tok(full_input_text, add_special_tokens=False).input_ids
        target_ids = self.tok(target_text, add_special_tokens=False).input_ids
        
        input_ids = (prompt_ids + target_ids + [self.tok.eos_token_id])[:self.max_length]
        
        # Masking the prompt so the model only learns the response
        labels = ([-100] * len(prompt_ids) + target_ids + [self.tok.eos_token_id])[:self.max_length]
        
        # Padding
        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tok.pad_token_id] * pad_len
        labels += [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor([1 if x != self.tok.pad_token_id else 0 for x in input_ids])
        }

# ... [Rest of the training loop from v10 remains the same] ...