# ATC LLM Agents: Agentic ATC Communication Prototype

<p align="center">
  <img src="assets/atc_agentic_prototype_pipeline.png" alt="ATC Agentic Prototype Pipeline" width="95%">
</p>

<p align="center">
  <b>XAION / ATC LLM agentic prototype for vocal input, ASR, transcript correction, Phi-4 ATC response generation, monitor validation, and text-to-speech output.</b>
</p>

---

## Overview

`ATC_LLM_AGENTS` extends the earlier ATC LLM work into an integrated agentic prototype for Air Traffic Control (ATC) communication research. The repository focuses on a speech-to-response pipeline where multiple components cooperate rather than treating the LLM as a single isolated text generator.

The prototype supports ATC-style interaction modes such as **Single Handoff**, **Vocal Input**, and **Full Simulation**. The system connects automatic speech recognition, text-to-text correction, Phi-4-based response generation, Phi-4-based monitoring, deterministic formatting checks, and text-to-speech output.

This work is part of the broader GenATC / XAION direction: using Digital Twin context, aircraft state, runway configuration, and ATC phraseology constraints to support predictive, context-aware, and safety-aware ATC communication modeling.

> **Research-use note:** This repository is a research prototype for ATC communication modeling and simulation. It is not intended for operational air traffic control use.

---

## ATC Agentic Prototype Pipeline

The architecture diagram summarizes the current agentic workflow:

1. **Vocal Input**  
   A pilot/user provides spoken input through the interface.

2. **ASR Agent — Whisper API**  
   The spoken input is converted into a raw pilot transcription.

3. **Pilot Transcription**  
   The initial ASR text is stored and passed to the correction stage.

4. **LLM Corrector — Phi-4 Text-to-Text Corrector**  
   The correction model refines ASR text to improve call sign consistency, aviation wording, and formatting.

5. **Corrected Transcription**  
   The corrected pilot line becomes the grounded text input for ATC response generation.

6. **LLM Responder — Phi-4 Responder**  
   The responder generates a concise ATC-style radio line using the corrected pilot message and available scenario or Digital Twin context.

7. **LLM Monitor — Phi-4 Monitor**  
   The monitor checks for repeated tokens, malformed phraseology, prompt leakage, turn inconsistency, unsafe wording, and missing operational details. It can approve, repair, or request regeneration.

8. **TTS Agent — ElevenLabs API**  
   The validated ATC response is converted into spoken audio.

9. **ATC-Safe Response Support**  
   The final response is presented as a controlled ATC-style output for simulation, demonstration, and evaluation.

---

## Project Goals

- Build an end-to-end voice-loop prototype for ATC communication support.
- Support pilot-initiated ATC interactions through vocal input.
- Improve raw ASR transcripts using LLM-based text-to-text correction.
- Generate concise, FAA-style ATC responses using Phi-4-based models.
- Add a Phi-4 monitor layer for agentic validation and response repair.
- Use deterministic formatting fixes for call signs, runway identifiers, altitudes, frequencies, and readback consistency.
- Incorporate Digital Twin context for simulation-grounded ATC decision support.
- Support anomaly-aware and context-aware ATC simulation experiments.

---

## Main Features

### End-to-End Voice Loop

- Vocal input support.
- Whisper-based automatic speech recognition.
- Pilot transcription capture.
- LLM-based transcript correction.
- Phi-4 ATC response generation.
- Phi-4 monitor validation.
- ElevenLabs text-to-speech output.

### Prototype Modes

| Mode | Description |
|---|---|
| Single Handoff | Simulates a short ATC handoff sequence with pilot and controller turns |
| Vocal Input | Allows the user to act as the pilot through speech input |
| Full Simulation | Runs longer pilot-controller interaction sequences through the prototype |
| Monitor + Context | Uses response monitoring and contextual information to improve response safety and consistency |

### Phi-4-Based Agent Roles

| Agent | Purpose |
|---|---|
| Phi-4 Corrector | Refines raw ASR transcription into cleaner aviation text |
| Phi-4 Responder | Generates the ATC/controller response |
| Phi-4 Monitor | Checks response quality, phraseology, safety, and turn consistency |

### Engineering Fixes

The prototype includes several post-generation safeguards:

- Prompt engineering for single-line, role-specific ATC responses.
- Runway-side specificity when left/right runway support exists.
- Monitor checks for prompt bleed, repeated tokens, and malformed phraseology.
- Deterministic cleanup for callsigns, runway names, altitudes, and frequencies.
- Response filtering to reduce non-radio narration or explanation text.

---

## Relationship to the Three-Stage ATC LLM Pipeline

This repository connects directly to the broader three-stage ATC LLM methodology:

| Stage | Purpose | Repository Connection |
|---|---|---|
| Stage 1: Domain Adaptation | Fine-tune models on FAA/ICAO aviation documents and QA pairs | Provides aviation procedural grounding |
| Stage 2: Phraseology-Aware Instruction Tuning | Train on structured ATC dialogue templates | Supports role consistency, sequencing, and FAA-style output |
| Stage 3: Operational Context Alignment | Re-tune with Digital Twin / KGSO simulation context | Enables context-aware ATC responses conditioned on aircraft state and operational setting |

The agentic prototype uses these model-development ideas in a deployed pipeline where speech, transcription, response generation, monitoring, and TTS are connected.

---

## Example Agentic Pipeline Interaction

The example below reflects the current direction of the prototype more accurately than a generic free-form conversation. It separates the speech/ASR/correction portion from the ATC response-generation and monitor-validation portion.

```text
Vocal Input / Pilot:
Greensboro Approach, DAL142, level six thousand feet, inbound to GSO.

ASR Output:
Greensboro Approach, DAL142, level 6000 feet, inbound to GSO.

Corrected Transcription:
Greensboro Approach, DAL142, level 6000 feet, inbound to GSO.

Scenario / DT Context:
Aircraft DAL142 is inbound to Greensboro at 6000 feet and is being sequenced for an ILS approach to Runway 23L with an assigned descent to 4,000 feet.

Phi-4 Responder:
DAL142, radar contact, descend and maintain 4,000, expect ILS Runway 23L.

Phi-4 Monitor:
Valid ATC line. The response preserves the call sign, altitude assignment, runway, and one-line FAA-style phraseology.

TTS Output:
The validated ATC response is converted to speech through ElevenLabs.
```

---

## Additional Stage-3 Context-Grounded Examples

These examples show how the responder can use synchronized Digital Twin context and a pilot transmission to produce an ATC-style response.

```text
Context:
[DT State]: ALT 800 ft; 5 NM final; RWY 05R active.
[DT Context]: Wind 230 at 10.
[Pilot]: Tower, N345GH, 5 mile final Runway 05R.

ATC Response:
N345GH, wind 230 at 10, Runway 05R cleared to land.
```

```text
Context:
[DT State]: On ground; taxi route available via Bravo and Charlie.
[DT Context]: Runway 23L active; crossing not authorized.
[Pilot]: Ground, AAL202, request taxi to Gate 3.

ATC Response:
AAL202, taxi to Gate 3 via Bravo, Charlie, hold short of Runway 23L.
```

---

## Repository Structure

```text
ATC_LLM_AGENTS/
│
├── README.md
│
├── XAION_CONTROL_main.py
│   └── Main prototype script for the agentic ATC communication workflow.
│
├── Generate_SyntheticCom_Phi_main.py
│   └── Generates synthetic ATC communications using Phi-based prompting.
│
├── phi4_synthetic_comm_metrics_ACCV1.csv
│   └── Metrics for synthetic ATC communication generation.
│
├── phi4_synthetic_conversation_metrics_PilotInitiated.csv
│   └── Metrics for pilot-initiated ATC conversation generation.
│
├── Anomalous Contextual Datasets/
│   └── Contextual anomaly datasets for ATC and digital-twin-style scenarios.
│
├── Global Prompt + Template example doc/
│   └── Prompt templates and example prompt framing.
│
├── Hand Off SOP ATC docs/
│   └── ATC handoff-related SOP material and supporting documents.
│
├── Main Datasets/
│   └── Core datasets used for ATC communication experiments.
│
├── phi4_stage2_instruction_variant_jsonl_csv/
│   └── Stage-2 instruction variant datasets in JSONL/CSV formats.
│
└── assets/
    └── atc_agentic_prototype_pipeline.png
```

---

## Installation

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install core dependencies:

```bash
pip install torch transformers pandas numpy scikit-learn gradio
pip install openai elevenlabs python-dotenv
```

If running local or API-based ASR, install the package used by your selected configuration:

```bash
pip install openai-whisper
```

Optional evaluation packages:

```bash
pip install nltk rouge-score bert-score editdistance sentence-transformers
```

---

## Configuration

Create a local environment file if API-based services are used:

```bash
touch .env
```

Example `.env` values:

```bash
OPENAI_API_KEY=your_openai_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

If your project uses a separate configuration file, place API keys, model paths, and voice IDs there instead of committing them to GitHub.

> Do not commit private API keys, tokens, model credentials, model checkpoints with restricted licenses, or personal audio data.

---

## How to Run

### 1. Run the Main Agentic ATC Prototype

```bash
python XAION_CONTROL_main.py
```

Depending on the local script configuration, the interface may launch as a terminal workflow or a Gradio-based prototype.

### 2. Run Synthetic ATC Communication Generation

```bash
python Generate_SyntheticCom_Phi_main.py
```

This script supports synthetic ATC phraseology generation and can be used to evaluate Phi-based ATC communication behavior.

---

## Example Workflow

```text
Pilot Voice Input
    ↓
Whisper ASR
    ↓
Raw Pilot Transcript
    ↓
Phi-4 Transcript Corrector
    ↓
Corrected Pilot Transcript
    ↓
Phi-4 ATC Responder
    ↓
Phi-4 Monitor / Agentic Validation
    ↓
Deterministic Formatting Checks
    ↓
ElevenLabs TTS
    ↓
ATC-Safe Spoken Response
```

---

## Evaluation Focus

The pipeline can be evaluated using both general NLP metrics and ATC-specific operational metrics.

### General Text Metrics

- BLEU
- ROUGE-L
- Cosine similarity
- BERTScore
- Edit distance
- Perplexity

### ATC-Specific Metrics

- Call Sign Accuracy
- Call Sign Word Error Rate
- Slot Error Rate
- Readback correctness
- Contextual compliance
- Phraseology consistency
- Handoff correctness
- Response conciseness
- Monitor pass/fail behavior
- Latency, memory, and precision trade-offs

---

## Dataset Categories

This repository is organized around several dataset types:

| Dataset Type | Description |
|---|---|
| Main ATC datasets | Core ATC dialogue and phraseology data |
| Stage-2 instruction variants | Prompt and response formats for phraseology-aware instruction tuning |
| Stage-3 contextual examples | Digital Twin / KGSO context paired with pilot transmissions and ATC responses |
| Anomalous contextual datasets | Scenario data for anomaly-aware ATC response modeling |
| Handoff SOP documents | Supporting material for ATC handoff behavior |
| Prompt/template examples | Global prompt structures and reusable templates |
| Synthetic communication metrics | Results and logs for generated ATC outputs |

---

## Suggested Image Setup

Place the pipeline image in the repository as:

```text
assets/atc_agentic_prototype_pipeline.png
```

Then the README image line will render correctly:

```html
<p align="center">
  <img src="assets/atc_agentic_prototype_pipeline.png" alt="ATC Agentic Prototype Pipeline" width="95%">
</p>
```

---

## Relationship to `ATC_LLM`

This repository builds on the earlier `ATC_LLM` work. The earlier repository focused on LLM fine-tuning, aviation QA evaluation, phraseology-aware instruction tuning, and simulation-grounded ATC response generation. `ATC_LLM_AGENTS` moves the work toward an integrated agentic prototype where speech, transcription correction, LLM response generation, response monitoring, deterministic cleanup, and TTS are connected in a single ATC communication workflow.

---

## Citation / Acknowledgment

If using this repository in a paper, poster, or presentation, please cite the project as an ATC LLM agentic prototype developed for research in phraseology-aware, simulation-grounded, and safety-aware air traffic control communication modeling.

---

## Disclaimer

This project is for research, prototyping, and simulation only. It does not provide certified aviation guidance and should not be used for real-world ATC operations, flight decision-making, or safety-critical deployment.
