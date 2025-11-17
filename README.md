# Cross-Doc App Runtime Layout

This tree captures the modules, scripts, and storage directories that the current
`app_crossdoc.py` Streamlit imports or writes to when bootstrapped with
`setup.sh`.  Optional assets (e.g., model checkpoints) are shown with placeholder
names so you can map your own files into the expected slots.

```
app_crossdoc.py
├── UI configuration & rendering helpers
│   ├── ui_config.py                  # default knobs (cache paths, sqlite name, model switches)
│   ├── st_helpers.py                 # sidebar + main-pane widgets, chip bars, query builders
│   └── ui_common.py                  # sentence overlay rendering utilities
│
├── Pipeline orchestration & runners
│   ├── bridge_runners.py             # ingestion, concordance, SciCo builders
│   ├── flexiconc_adapter.py          # SQLite persistence (sentences, tries, FAISS payloads)
│   ├── global_coref_helper.py        # trie & Jaccard scoring for cross-document chains
│   ├── helper_addons.py              # trie/co-occurrence builders for optional backends
│   └── cooc_utils.py                 # resolves spaCy/HF co-occurrence engines
│
├── Embedding + classifier helpers
│   ├── helper.py                     # MPNet, SDG-BERT, SciCo encoders + HF cache wiring
│   └── helper_addons.py              # (see above) exposes build_ngram_trie/build_cooc_graph
│
├── Upload & storage helpers
│   └── utils_upload.py               # persists uploaded PDFs to /tmp or storage/uploads
│
├── Reverse-attribution integrations
│   └── run_reverse_attribution_pipeline.py  # optional batch pipeline invoked via sidebar actions
│
├── Default data artifacts (created or referenced)
│   ├── flexiconc.sqlite              # default corpus database (configurable via sidebar)
│   ├── data/
│   │   ├── reverse_attribution/      # populated by scripts/download_legacy_models.py
│   │   │   └── rev_attr_summ_pdfs/...   # Google Drive PDF bundle (unzipped)
│   │   └── flexiconc/
│   │       ├── raw/                  # source PDFs / exports (user-managed)
│   │       ├── processed/            # ingested SQLite snapshots, JSONL, etc.
│   │       └── faiss/                # optional ANN shards when exporting outside SQLite
│   ├── storage/
│   │   └── uploads/                  # persistent upload directory created by setup.sh
│   └── outputs/                      # export target for production artifacts (graphs, CSVs)
│
├── Model checkpoints & caches
│   ├── models/
│   │   ├── presumm/
│   │   │   ├── model_step_30000.pt        # PreSumm extractive checkpoint
│   │   │   └── bert_abs_ext/...           # PreSumm abstractive archive contents
│   │   └── spanbert_coref/ (optional)     # SpanBERT/fastcoref checkpoints if cached locally
│   ├── BERT-KPE/ (optional)          # external repo for keyphrase extraction
│   │   └── checkpoints/bert2span.bin
│   └── .cache/huggingface/           # HF transformers cache (HF_CACHE_DIR)
│       └── models--*/datasets--*     # warmed by scripts/cache_hf_models.py
│
├── Bootstrap & maintenance scripts
│   ├── setup.sh                      # creates directories, installs deps, downloads assets
│   ├── scripts/
│   │   ├── download_legacy_models.py # grabs PreSumm + reverse-attribution bundles (gdown)
│   │   └── cache_hf_models.py        # preloads SDG classifier + encoder models into HF cache
│   └── make_flexiconc_db.py          # CLI for FlexiConc ingestion (used by bridge_runners)
│
└── Tests & supporting resources
    ├── test_flexiconc_adapter.py     # ensures DB ingest/export stays compatible
    ├── test_reverse_attribution.py   # regression coverage for attribution pipeline
    ├── justification.csv             # label descriptions surfaced in the UI
    └── doi_to_citation.json          # SciCo metadata enrichment
```

The tree highlights where to place your local assets so the Streamlit app can run
fully offline: run `./setup.sh` once, copy any existing `flexiconc.sqlite` into the
project root (or adjust the sidebar path), and drop additional model checkpoints
into the indicated directories if you rely on custom fine-tunes.