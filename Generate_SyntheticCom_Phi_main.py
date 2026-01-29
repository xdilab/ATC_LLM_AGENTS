import os
import csv
import torch
import random
import gc
import math
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Setup ===
nltk.download("punkt")
torch.cuda.empty_cache()
gc.collect()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "microsoft/phi-4"  # Replace if using fine-tuned path

# === Load Model and Tokenizer ===
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

# === Template Tags ===
templates = [
    "Greensboro Approach, <FH>, level <ALT> feet, inbound to GSO.",
    "<FH>, radar contact, descend and maintain <NEW_ALT>, expect ILS Runway <RWY>.",
    "<FH>, turn heading <HDG>, descend to <NEW_ALT>.",
    "<FH>, contact Greensboro Tower on <TF>.",
    "Greensboro Tower, <FH>, ILS <RWY>, <DIST> miles out.",
    "<FH>, wind <WIND>, Runway <RWY> cleared to land.",
    "<FH>, exit left on <TXWY>, contact Ground on <GF>.",
    "<FH>, taxi to <DEST> via <ROUTE>."
]

tag_values = {
    "<FH>": ["N466C7", "N123AB", "N345GH", "AAL202", "DAL338", "SWA519"],
    "<AF>": ["124.35", "126.75", "127.9"],
    "<TF>": ["118.5", "119.7", "120.1"],
    "<GF>": ["121.9", "122.35", "121.7"],
    "<ALT>": ["8,000", "7,000", "6,500", "10,000"],
    "<NEW_ALT>": ["3,000", "2,500", "4,000", "5,000"],
    "<RWY>": ["23L", "23R", "14", "5"],
    "<HDG>": ["270", "180", "90", "360"],
    "<DIST>": ["8", "10", "5", "3"],
    "<WIND>": ["230 at 10", "250 at 15", "180 at 5"],
    "<TXWY>": ["Bravo", "Charlie", "Delta"],
    "<DEST>": ["Signature", "Atlantic", "Gate 3"],
    "<ROUTE>": ["Bravo, Charlie, hold short Runway 23L", "Alpha, Echo", "Charlie, hold short 23R"]
}

# === Utilities ===
def fill_template(template):
    for tag in tag_values:
        if tag in template:
            template = template.replace(tag, random.choice(tag_values[tag]))
    return template

def extract_callsign(text):
    for call in tag_values["<FH>"]:
        if call in text:
            return call
    return None

def compute_call_accuracy(prompt, response):
    return int(extract_callsign(prompt) == extract_callsign(response))

def compute_call_wer(prompt, response):
    ref = extract_callsign(prompt)
    hyp = extract_callsign(response)
    if ref is None or hyp is None:
        return 1.0
    return 0.0 if ref == hyp else 1.0

def compute_edit_distance(a, b):
    return nltk.edit_distance(a, b)

def compute_bleu(prompt, generated):
    prompt_tokens = tokenizer.tokenize(prompt)
    generated_tokens = tokenizer.tokenize(generated)
    smoothie = SmoothingFunction().method4
    return sentence_bleu([prompt_tokens], generated_tokens, smoothing_function=smoothie)

rouge = Rouge()

def compute_rouge(prompt, response):
    scores = rouge.get_scores(response, prompt)
    return scores[0]["rouge-l"]["f"]

def compute_perplexity(text):
    input_ids = tokenizer(text, return_tensors="pt", padding=True).input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())

# === Generation + Evaluation ===
print("\n🛠️ Generating synthetic ATC communications...")
generated_data = []

for _ in range(100):
    template = random.choice(templates)
    filled_prompt = fill_template(template)

    inputs = tokenizer(filled_prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=64,
            num_beams=5,
            no_repeat_ngram_size=3,
            length_penalty=0.8,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    bleu = compute_bleu(filled_prompt, decoded)
    rouge_l = compute_rouge(filled_prompt, decoded)
    edit_dist = compute_edit_distance(filled_prompt, decoded)
    perplexity = compute_perplexity(decoded)
    call_acc = compute_call_accuracy(filled_prompt, decoded)
    call_wer = compute_call_wer(filled_prompt, decoded)

    generated_data.append([
        filled_prompt,
        decoded,
        round(call_acc, 3),
        round(call_wer, 3),
        round(bleu, 3),
        round(rouge_l, 3),
        round(edit_dist, 2),
        round(perplexity, 2)
    ])

# === Save Output ===
output_csv = "/home/smgreen1/PycharmProjects/GEN_ATC_LLM_Scripts/phi4_synthetic_comm_metrics_ACCV1.csv"
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Prompt", "Generated Response",
        "Call Sign Accuracy", "CallWER",
        "BLEU", "ROUGE-L",
        "Edit Distance", "Perplexity"
    ])
    writer.writerows(generated_data)

print(f"\n✅ Saved evaluated results to: {output_csv}")
