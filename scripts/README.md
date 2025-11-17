## ⚙️ 1. Overview

There are two main setup scripts:

| Script | Description |
|--------|--------------|
| `setup.sh` | Primary setup: installs dependencies, prepares cache directories, and fetches legacy Hugging Face models. |
| `legacy_models.sh` | Optional extended setup: installs and verifies **PreSumm (XSUM)** and **BERT-KPE** checkpoints from Google Drive. |

Both scripts are idempotent — rerunning them will skip already-downloaded components.

---

## 🧠 2. Directory Layout

After running the setup, the directory structure will look like this:
project_root/
├── .env                       # Environment variables for local runs
├── setup.sh                   # Main setup script
├── legacy_models.sh            # Legacy model bootstrap (PreSumm + KPE)
├── requirements.txt
├── models/
│   ├── presumm/                # PreSumm model checkpoint
│   └── bert_kpe/               # BERT-KPE checkpoint (unzipped)
├── data/
│   └── reverse_attribution/    # Placeholder data directory
├── outputs/                    # Experiment output
├── storage/uploads/            # Uploaded file cache
└── .cache/huggingface/         # Hugging Face model/dataset cache

---
## 🚀 3. Usage

### A. Standard Environment Setup

Run once to prepare dependencies and cache:

```bash
!bash setup.sh

To activate environment variables for interactive shells:
export $(grep -v '^#' .env | xargs)


B. PreSumm + BERT-KPE Setup
Run the legacy model bootstrap:
!bash legacy_models.sh

This will:


Clone the PreSumm repository.


Download the XSUM extractive summarization checkpoint from Google Drive (1TWuIJtIg9EHq3K-jFPVgt11JsiZstlxJ).


Validate model loading via xsum_rank.load_presumm_model.


Clone BERT-KPE and download its checkpoint archive (13FvONBTM4NZZCR-I7LVypkFa0xihxWnM).


Unzip and register paths in .env.


⚡️ 4. Custom Drive IDs
You can override checkpoint locations dynamically:
!PRESUMM_GDRIVE_ID=your_presumm_id \
 KPE_GDRIVE_ID=your_kpe_id \
 bash legacy_models.sh

or provide full Drive URLs:
!KPE_GDRIVE_URL="https://drive.google.com/file/d/1xxxxxx/view?usp=sharing" bash legacy_models.sh


🧩 5. Environment Variables
After setup, your .env file will contain:
# PreSumm
PRESUMM_DIR=/path/to/PreSumm
PRESUMM_MODELS_DIR=/path/to/models/presumm
PRESUMM_CKPT_PATH=/path/to/models/presumm/model_step_30000.pt

# BERT-KPE
KPE_DIR=/path/to/BERT-KPE
KPE_MODELS_DIR=/path/to/models/bert_kpe
KPE_CKPT_PATH=/path/to/models/bert_kpe/bert_kpe.ckpt

You can load these in Python:
import os
ckpt_presumm = os.getenv("PRESUMM_CKPT_PATH")
ckpt_kpe = os.getenv("KPE_CKPT_PATH")


🧪 6. Quick Verification
After both setups, verify everything loads properly:
from xsum_rank import load_presumm_model
from pathlib import Path
import torch

repo = os.getenv("PRESUMM_DIR")
ckpt = os.getenv("PRESUMM_CKPT_PATH")
device = "cuda" if torch.cuda.is_available() else "cpu"

model, tok = load_presumm_model(repo_dir=repo, ckpt_path=ckpt, device_str=device)
print("✅ PreSumm initialized successfully.")


🩵 7. Notes


Both scripts tolerate network or permission errors (Drive quota, etc.) and continue setup.


You can manually drop pretrained checkpoints into the corresponding directories if downloads fail.


For headless servers, run:
bash setup.sh --no-input


Authors:
Pawel Oczkowski et al.
(Vrije Universiteit Amsterdam / ENLENS Project)
2025
---


