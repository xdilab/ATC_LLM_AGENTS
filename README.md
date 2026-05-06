# ATC LLM Agents: Agentic Air Traffic Control Communication Prototype

<p align="center">
  <img src="assets/atc_agentic_prototype_pipeline.png" alt="ATC Agentic Prototype Pipeline" width="95%">
</p>

<p align="center">
  <b>Agentic ATC pipeline for speech input, transcription correction, LLM-based controller response generation, response monitoring, and voice output.</b>
</p>

---

## Overview

`ATC_LLM_AGENTS` extends the earlier ATC LLM work into an agentic prototype pipeline for Air Traffic Control (ATC) communication support. Instead of treating the model as a single text generator, this repository organizes the workflow into coordinated modules that handle speech input, automatic speech recognition, transcript correction, ATC response generation, response monitoring, and text-to-speech output.

The prototype is designed around realistic pilot-controller communication flows, including handoff scenarios, phraseology-aware responses, and ATC-safe response support. The pipeline supports experimentation with Phi-4-based ATC models, Whisper-based ASR, ElevenLabs TTS, synthetic ATC datasets, and contextual/anomalous aviation scenarios.

> **Research-use note:** This repository is a research prototype for ATC communication modeling and simulation. It is not intended for operational air traffic control use.

---

## ATC Agentic Prototype Pipeline

The attached architecture diagram illustrates the full prototype workflow:

1. **Vocal Input**  
   Pilot speech or user-provided audio is used as the entry point into the system.

2. **ASR Agent — Whisper API**  
   The audio input is transcribed into raw pilot text.

3. **Pilot Transcription**  
   The first-pass transcript is stored and passed forward for correction.

4. **LLM Corrector — Phi-4 Text-to-Text Corrector**  
   A Phi-4-based correction model refines the raw ASR transcript to improve aviation phraseology, call sign consistency, and readability.

5. **Corrected Transcription**  
   The corrected pilot message becomes the grounded input for response generation.

6. **LLM Responder — Phi-4 Responder**  
   A Phi-4-based ATC response model generates the controller response.

7. **LLM Monitor — Phi-4 Monitor**  
   A monitoring model checks or guides the generated response before it is converted to speech. This supports safer, more controlled ATC-style communication.

8. **TTS Agent — ElevenLabs API**  
   The approved response is converted into voice output.

9. **ATC-Safe Response Support**  
   The final system output is a controlled ATC-style response intended for simulation, training, and research evaluation.

---

## Project Goals

- Build an agentic ATC communication pipeline around speech, text, LLM response generation, monitoring, and TTS.
- Support pilot-initiated and controller-response dialogue generation.
- Improve ASR outputs using LLM-based transcript correction.
- Generate concise, phraseology-aware ATC responses using Phi-4-based models.
- Add a monitoring layer to reduce unsafe, verbose, or off-topic generations.
- Support handoff, approach, tower, ground, and synthetic ATC communication scenarios.
- Enable experimentation with anomaly-aware and context-aware ATC datasets.

---

## Main Features

### Speech-to-Response ATC Pipeline

- Vocal input support.
- Whisper-based automatic speech recognition.
- Pilot transcription capture.
- LLM-based transcript correction.
- ATC response generation.
- LLM-based response monitoring.
- ElevenLabs text-to-speech integration.

### Phi-4-Based ATC Agents

This repository centers on several Phi-4-based roles:

| Agent | Purpose |
|---|---|
| Phi-4 Corrector | Refines raw ASR transcription into cleaner aviation text |
| Phi-4 Responder | Generates the ATC/controller response |
| Phi-4 Monitor | Checks response quality, safety, and turn-taking behavior |

### ATC Communication Scenarios

The repository supports experimentation with:

- Pilot check-in.
- ATC handoff.
- Approach and tower communication.
- Ground/taxi instructions.
- Takeoff and landing-related dialogue.
- Pilot-initiated conversations.
- Contextual anomaly scenarios.
- Synthetic ATC communication generation.

### Dataset and Evaluation Support

The repository includes or supports:

- Main ATC datasets.
- Anomalous contextual datasets.
- Stage-2 instruction variant datasets.
- Global prompt and template examples.
- Hand-off SOP ATC documents.
- Synthetic communication metrics.
- Pilot-initiated conversation metrics.

---

## Repository Structure

```text
ATC_LLM_AGENTS/
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

> Do not commit private API keys, tokens, model credentials, or personal audio data.

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
Phi-4 Monitor / Safety Check
    ↓
ElevenLabs TTS
    ↓
ATC-Safe Spoken Response
```

---

## Example Pilot-to-ATC Interaction

```text
Pilot:
Greensboro Approach, N123AB, level 4,000, inbound for the ILS Runway 23.

Corrected Transcription:
Greensboro Approach, November One Two Three Alpha Bravo, level four thousand, inbound for the ILS Runway Two Three.

ATC Response:
November One Two Three Alpha Bravo, Greensboro Approach, radar contact. Proceed direct final approach course, maintain four thousand until established, cleared ILS Runway Two Three approach.

Monitor Decision:
Response follows expected ATC phraseology and includes call sign, facility, altitude constraint, and approach clearance.

TTS Output:
Spoken ATC response generated through ElevenLabs.
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
- Slot F1
- Phraseology consistency
- Handoff correctness
- Response conciseness
- Safety-monitor pass/fail behavior

---

## Dataset Categories

This repository is organized around several dataset types:

| Dataset Type | Description |
|---|---|
| Main ATC datasets | Core ATC dialogue and phraseology data |
| Stage-2 instruction variants | Prompt and response formats for instruction tuning |
| Anomalous contextual datasets | Scenario data for anomaly-aware ATC response modeling |
| Handoff SOP documents | Supporting material for ATC handoff behavior |
| Prompt/template examples | Global prompt structures and reusable templates |
| Synthetic communication metrics | Results and logs for generated ATC outputs |

---

## Relationship to `ATC_LLM`

This repository builds on the earlier `ATC_LLM` work. The earlier repository focused heavily on LLM fine-tuning, aviation QA evaluation, and synthetic ATC communication generation. `ATC_LLM_AGENTS` moves the work toward an integrated, agentic prototype where multiple components cooperate in a speech-to-response ATC communication pipeline.

---

## Suggested Image Setup

Place the attached pipeline image in the repository as:

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

## Citation / Acknowledgment

If using this repository in a paper, poster, or presentation, please cite the project as an ATC LLM agentic prototype developed for research in phraseology-aware and safety-aware air traffic control communication modeling.

---

## Disclaimer

This project is for research, prototyping, and simulation only. It does not provide certified aviation guidance and should not be used for real-world ATC operations, flight decision-making, or safety-critical deployment.
