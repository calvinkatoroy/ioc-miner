# ioc-miner

Threat intelligence IOC extractor combining regex patterns, SecBERT NER, and context classification to pull indicators from threat reports, blogs, PDFs, and social feeds.

## Features

- **Dual extraction**: fast regex baseline + optional fine-tuned SecBERT NER model
- **Context classification**: heuristic + optional zero-shot NLI to verdict each IOC (malicious / benign / sinkhole / unknown)
- **Multi-source ingestion**: PDF, HTML, plaintext, Telegram channels, Twitter/X
- **Multiple output formats**: STIX 2.1, MISP JSON, CSV, rich terminal table
- **Defang-aware**: handles `hxxp://`, `[.]`, `(dot)` variants

## IOC Types Extracted

| Type | Examples |
|------|---------|
| IP | `1.2.3.4` |
| Domain | `evil.example.ru` |
| URL | `https://c2.evil.com/beacon` |
| MD5 / SHA1 / SHA256 / SHA512 | file hashes |
| CVE | `CVE-2024-1234` |
| Email | `attacker@domain.com` |
| Filepath | `C:\Windows\Temp\payload.exe` |

## Installation

```bash
pip install -e .

# Optional extras
pip install -e ".[telegram]"   # Telegram ingestion
pip install -e ".[twitter]"    # Twitter/X ingestion
pip install -e ".[train]"      # Fine-tuning dependencies
pip install -e ".[dev]"        # Dev/test tools
```

Requires Python 3.10+.

## Usage

```bash
# Extract from PDF, output STIX
ioc-miner extract report.pdf --format stix

# Extract from URL, drop benign IOCs, save to file
ioc-miner extract https://blog.example.com/threat-report --format csv --no-benign --output iocs.csv

# Pipe plaintext via stdin
cat report.txt | ioc-miner extract - --format stdout

# Use fine-tuned SecBERT NER model
ioc-miner extract report.pdf --model ./models/secbert-ner --format stix

# Enable NLI classifier for ambiguous IOCs
ioc-miner extract report.html --use-ml --no-unknown

# Telegram channel (requires Telethon session)
ioc-miner extract @channelname --source telegram --tg-limit 500
```

### Options

| Flag | Description |
|------|-------------|
| `--format` / `-f` | Output format: `stix`, `csv`, `misp`, `stdout` (default: `stdout`) |
| `--source` / `-s` | Force source type: `auto`, `pdf`, `html`, `text`, `telegram`, `twitter` |
| `--output` / `-o` | Write to file instead of stdout |
| `--model` / `-m` | HuggingFace model ID or local path to enable NER extraction |
| `--use-ml` | Enable zero-shot NLI classifier for ambiguous IOCs |
| `--no-benign` | Exclude benign/sinkhole IOCs |
| `--no-unknown` | Exclude IOCs with unknown verdict |
| `--tg-limit` | Max Telegram messages to fetch (default: 200) |
| `--tw-max` | Max tweets to fetch (default: 100) |
| `--quiet` / `-q` | Suppress progress output |

## Fine-tuning SecBERT

A Colab-ready notebook and training scripts are in [training/](training/).

```bash
# Prepare dataset from annotated JSONL
python training/prepare_dataset.py --input data/annotated.jsonl --output data/hf_dataset

# Fine-tune locally
python training/fine_tune_secbert.py --dataset data/hf_dataset --output models/secbert-ner
```

Or open [training/colab_finetune.ipynb](training/colab_finetune.ipynb) in Google Colab for GPU-accelerated training.

## Project Structure

```
ioc_miner/
├── ingestion/        # Source readers (PDF, HTML, plaintext, Telegram, Twitter)
├── preprocessing/    # Text normalization, defang reversal, sentence tokenization
├── extraction/       # RegexExtractor, NERExtractor, ContextClassifier
├── models/           # IOC dataclass and enums
└── output/           # STIX, MISP, CSV formatters
training/             # Fine-tuning scripts and Colab notebook
tests/                # Pytest test suite
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```
