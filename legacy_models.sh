# --------------- PreSumm (XSUM) setup ---------------
echo "⚙️ Setting up PreSumm XSUM checkpoint ..."

python -m pip install -q --upgrade gdown gitpython

# Define paths and constants
PRESUMM_REPO_URL="https://github.com/nlpyang/PreSumm"
PRESUMM_DIR="${PRESUMM_DIR:-$ROOT_DIR/PreSumm}"
PRESUMM_MODELS_DIR="${PRESUMM_MODELS_DIR:-$ROOT_DIR/models/presumm}"
mkdir -p "$PRESUMM_MODELS_DIR"

# Your Drive file ID
GDRIVE_ID="${PRESUMM_GDRIVE_ID:-1TWuIJtIg9EHq3K-jFPVgt11JsiZstlxJ}"
CKPT_PATH="$PRESUMM_MODELS_DIR/model_step_30000.pt"

# Clone repo if missing
if [[ ! -d "$PRESUMM_DIR/.git" ]]; then
  echo "🔹 Cloning PreSumm repository ..."
  git clone --depth 1 "$PRESUMM_REPO_URL" "$PRESUMM_DIR" || echo "⚠️ Clone failed; continuing."
else
  echo "✅ PreSumm repo already present at $PRESUMM_DIR"
fi

# Download the checkpoint from Drive if not already there
if [[ ! -f "$CKPT_PATH" ]]; then
  echo "⬇️  Downloading PreSumm checkpoint from Google Drive..."
  set +e
  gdown --id "$GDRIVE_ID" -O "$CKPT_PATH"
  DL_STATUS=$?
  set -e
  if [[ $DL_STATUS -ne 0 || ! -f "$CKPT_PATH" ]]; then
    echo "⚠️ Could not download PreSumm checkpoint from Drive."
    echo "   Verify access or manually place it at: $CKPT_PATH"
  else
    echo "✅ Checkpoint saved to $CKPT_PATH"
  fi
else
  echo "✅ Checkpoint already present at $CKPT_PATH"
fi

# Store for later runs
{
  echo "PRESUMM_DIR=$PRESUMM_DIR"
  echo "PRESUMM_MODELS_DIR=$PRESUMM_MODELS_DIR"
  echo "PRESUMM_CKPT_PATH=$CKPT_PATH"
} >> "$ROOT_DIR/.env"

# Try loading to confirm it works (non-fatal)
python - <<'PY'
import os, sys, torch
repo_dir = os.environ.get("PRESUMM_DIR")
ckpt = os.environ.get("PRESUMM_CKPT_PATH")
device = "cuda" if torch.cuda.is_available() else "cpu"

if repo_dir and os.path.isdir(repo_dir) and repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

print(f"🔍 Checking PreSumm model from {ckpt} on device {device}")
try:
    from xsum_rank import load_presumm_model
    model, tok = load_presumm_model(
        repo_dir=repo_dir,
        ckpt_path=ckpt,
        device_str=device,
        repo_url="https://github.com/nlpyang/PreSumm"
    )
    print("✅ PreSumm model loaded successfully.")
except Exception as e:
    print(f"⚠️ PreSumm load skipped or failed: {e}")
PY

# --------------- BERT-KPE bootstrap (non-fatal) ---------------
if [[ -f "$ROOT_DIR/get_bert_kpe.py" ]]; then
  echo "🚀 Setting up BERT-KPE (checkpoint from Drive)..."
  python -m pip install -q --upgrade gdown
  set +e
  # Provide either KPE_GDRIVE_URL or KPE_GDRIVE_ID; the script supports both.
  KPE_DIR="${KPE_DIR:-$ROOT_DIR/BERT-KPE}" \
  KPE_GDRIVE_ID="${KPE_GDRIVE_ID:-13FvONBTM4NZZCR-I7LVypkFa0xihxWnM}" \
  python "$ROOT_DIR/get_bert_kpe.py"
  KPE_STATUS=$?
  set -e
  if [[ $KPE_STATUS -ne 0 ]]; then
    echo "⚠️ BERT-KPE bootstrap encountered an error; continuing."
  else
    echo "✅ BERT-KPE ready."
  fi

  # Persist to .env
  {
    echo "KPE_DIR=${KPE_DIR:-$ROOT_DIR/BERT-KPE}"
  } >> "$ROOT_DIR/.env"
else
  echo "ℹ️ get_bert_kpe.py not found; skipping KPE setup."
fi
