# MHL-AML: Memory Hygiene Layer — Defense Against Adversarial Memory Injection Attacks on RAG-Based AML Agents

[![SSRN](https://img.shields.io/badge/SSRN-6734225-blue.svg)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6734225)
[![ACM ICAIF 2026](https://img.shields.io/badge/ACM%20ICAIF-2026%20Under%20Review-orange.svg)](https://icaif2026.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--3483--0025-green.svg)](https://orcid.org/0009-0008-3483-0025)

> **Paper:** *Poisoning the Compliance Mind: Adversarial Memory Injection Attacks on RAG-Based AML Agents*
> **Author:** Frankline Ondieki Ombachi
> **Preprint:** [SSRN 6734225](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6734225) · **Under peer review at ACM ICAIF 2026, Milan**

---

## The Problem in One Paragraph

RAG-based AI agents are being deployed at financial institutions globally to automate anti-money laundering compliance. These agents retrieve documents from a persistent vector memory at inference time to make risk decisions. We demonstrate that an adversary who controls as few as **50 documents** fed into this memory — through entirely legitimate ingestion pathways — can collapse the agent's detection accuracy from **91.9% to 1.4%** with **zero anomalous signatures** in any operational log. We call this the **Compliance Memory Poisoning Problem (CMPP)**. We also design and implement a defense: the **Memory Hygiene Layer (MHL)**.

---

## Key Results

| Metric | Result |
|---|---|
| Baseline AML agent F1 | **0.919** |
| After QTPI attack at β=50 documents | **F1 → 0.014** — 98.4% degradation |
| Attack log footprint | **Zero anomalous signatures** |
| MHL CPT detection rate | **100%** adversarial documents detected |
| MHL false quarantine rate | **0%** |
| MHL FCS latency p99 | **22.6 ms** on CPU — no GPU required |
| Minimum effective injection budget | **50 documents** |

---

## Three Attack Vectors

```
Adversary Goal: Suppress SAR generation for target AML typology
─────────────────────────────────────────────────────────────────
V1 — Document Poisoning (DP)
     Inject plausible documents mischaracterising typology as low-risk
     Result: F1 remains HIGH (0.924–0.937) — heuristic classifier resistant

V2 — Query-Time Prompt Injection (QTPI)  ← MOST LETHAL
     Embed adversarial instruction sequences in retrieved documents
     Result: F1 collapses to 0.007–0.017 across ALL tested budgets
     Mechanism: Hijacks agent instruction-following at inference time

V3 — Retrieval Rank Manipulation (RRM)
     Optimise documents for high cosine similarity with target embeddings
     Result: F1 remains HIGH (0.927–0.943) — primacy bias insufficient alone
```

---

## The Memory Hygiene Layer (MHL)

A modular, training-free defense that intercepts documents before they enter the vector index.

```
Incoming Document
       │
       ▼
┌─────────────────────────────────┐
│  Module 1: CPT                  │
│  Cryptographic Provenance       │
│  Tracking                       │
│  · SHA-256 content hashing      │
│  · Source trust scoring         │
│  · Provenance audit trail       │
│  Result: 100% detection, 0% FQ  │
└──────────────┬──────────────────┘
               │ PASS
               ▼
┌─────────────────────────────────┐
│  Module 2: FCS                  │
│  Factual Consistency Scoring    │
│  · Cross-encoder re-ranker      │
│  · Anchor corpus comparison     │
│  · 22.6ms p99 latency on CPU    │
└──────────────┬──────────────────┘
               │ PASS
               ▼
┌─────────────────────────────────┐
│  Module 3: SRAD                 │
│  Statistical Retrieval          │
│  Anomaly Detection              │
│  · KS-test on retrieval dist.   │
│  · 500-query sliding window     │
└──────────────┬──────────────────┘
               │ PASS
               ▼
        Vector Index ✓
```

---

## Experimental Results

### Attack degradation — TBML typology

| Attack Vector | β=50 | β=100 | β=280 | β=500 | β=1000 |
|---|---|---|---|---|---|
| Baseline (no attack) | 0.919 | 0.919 | 0.919 | 0.919 | 0.919 |
| DP — Document Poisoning | 0.924 | 0.930 | 0.934 | 0.936 | 0.937 |
| **QTPI — Prompt Injection** | **0.014 ⚠** | **0.014** | **0.010** | **0.007** | **0.017** |
| RRM — Rank Manipulation | 0.942 | 0.938 | 0.943 | 0.927 | 0.927 |

> ⚠ QTPI achieves 98.4% degradation at β=50 — the minimum tested budget. The agent becomes operationally equivalent to random guessing while producing no anomalous log signatures.

### MHL defense results — β=280 DP attack

| Configuration | F1 | Detection Rate | False Quarantine | Latency p99 |
|---|---|---|---|---|
| No defense | 0.945 | — | — | 0 ms |
| **CPT only** | **0.925** | **100%** | **0%** | **0 ms** |
| FCS only (θ=0.65) | 0.939 | 0% | 59.8% | 22.6 ms |
| SRAD only | 0.926 | 0% | 0% | 0 ms |
| MHL Full (θ=0.55) | 0.935 | 0% | 30.4% | 22.3 ms |
| MHL Full (θ=0.65) | 0.928 | 0% | 62.5% | 21.0 ms |

---

## Quickstart

### Option A: Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `run_experiments.py`
3. Run:

```python
# Install dependencies
!pip install faiss-cpu sentence-transformers openai pandas numpy scikit-learn tqdm -q
```

```python
# Run all experiments
import os
os.environ["OPENAI_API_KEY"] = ""  # Leave blank for simulation mode

exec(open('run_experiments.py').read())
run_all_experiments()
```

```python
# Download results
import shutil
from google.colab import files
shutil.make_archive('results', 'zip', 'results')
files.download('results.zip')
```

### Option B: Local

```bash
git clone https://github.com/OndiekiFrank/MHL-AML.git
cd MHL-AML
pip install -r requirements.txt
python run_experiments.py
```

> **Note:** Set `OPENAI_API_KEY` for full LLM-backed results. Leave unset to run in heuristic simulation mode — all attack and defense results are reproducible without an API key.

---

## Repository Structure

```
MHL-AML/
├── run_experiments.py      # Complete experiment runner
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── results/                # Output directory (generated on run)
    ├── degradation_curves.csv
    ├── mhl_defense.csv
    ├── typology_breakdown.csv
    └── summary.txt
```

---

## Regulatory Mapping

| Regulation | CMPP Vulnerability | MHL Mitigation |
|---|---|---|
| FINRA 2026 Rule 3110 | QTPI corrupts agent reasoning with no anomalous log signature | CPT audit trail maps every ingestion decision with cryptographic integrity |
| FCA Traceability Standards | Poisoned retrievals produce locally coherent but untraceable decisions | MHL links every decision to retrieved context, source provenance, and FCS score |
| FinCEN 2024 Proposed Rule | Agent at F1=0.014 is not "reasonably designed" for adversarial environments | CPT + SRAD provide proportionate, auditable, deployable controls |
| FATF 2025 AI Horizon Scan | LLM-generated adversarial documents bypass human plausibility review | CPT provenance tracking detects source-level anomalies below lexical surface |
| NIST AI RMF | No current profile governs retrieval corpus integrity | MHL provides first reference implementation for RAG corpus security controls |
| OWASP LLM Top 10 | Prompt injection (LLM01) extended to corpus-level — not covered | QTPI formalises a new sub-category requiring addition to future OWASP guidance |

---

## Why This Matters

Current AI security frameworks — NIST AI RMF, OWASP LLM Top 10, MITRE ATLAS — do not govern the retrieval corpus layer of deployed RAG systems. This is the gap this research addresses.

**The attack surface:** Every RAG-based AI agent that ingests documents from external or semi-trusted sources is potentially vulnerable to CMPP. This includes AI compliance agents, RAG-based legal research tools, AI-powered due diligence systems, and automated regulatory reporting agents.

**The defense:** MHL is designed to be deployed as a pre-ingestion gate in any existing RAG pipeline without model retraining, infrastructure changes, or GPU requirements.

---

## Citation

```bibtex
@article{ondieki2026poisoning,
  title   = {Poisoning the Compliance Mind: Adversarial Memory Injection Attacks on
             RAG-Based AML Agents},
  author  = {Ondieki Ombachi, Frankline},
  journal = {SSRN Preprint 6734225},
  year    = {2026},
  url     = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6734225},
  note    = {Under peer review at ACM ICAIF 2026, Milan, Italy}
}
```

---

## Author

**Frankline Ondieki Ombachi**
AI Security Researcher | Machine Learning Engineer | Nairobi, Kenya

📧 ondiekifrank021@gmail.com
🔗 [GitHub](https://github.com/OndiekiFrank)
📄 [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6734225)
🔬 [ORCID: 0009-0008-3483-0025](https://orcid.org/0009-0008-3483-0025)

- 🎓 Incoming Cyber Security Analytics student — Mohawk College, Hamilton, Ontario, Canada (September 2026)
- 🏆 Top 100 Rising AI Developers in Africa 2025 — UNDP, African Development Bank, Microsoft, Meta
- 🎤 Talk submitted — BSides Toronto 2026, October 3

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Disclaimer

The attack methodology is provided for academic reproducibility and defensive research purposes only. The Memory Hygiene Layer defense is described in greater implementation detail than the attacks, reflecting the net-positive safety intent of this disclosure. The authors do not condone malicious use of these techniques.
