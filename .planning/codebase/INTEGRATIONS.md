# External Integrations

## APIs & External Services

### HuggingFace Hub

**Purpose:** DNA sequence model (DNABERT-S)

**Model:** `zhihan1996/DNABERT-S`
- DNA language model for sequence embedding
- Accessed via `transformers` library
- Downloads model weights on first use

**Authentication:** None required (public model)

**Usage Location:**
- `virnucpro/pipeline/feature_extraction.py`
- Loaded via `AutoTokenizer` and `AutoModel`

**Configuration:**
- Model name in `config/default_config.yaml` (implied)
- `trust_remote_code=True` flag used

### Facebook Research ESM

**Purpose:** Protein sequence model (ESM-2 3B)

**Model:** ESM-2 3B parameter model
- Protein language model for sequence embedding
- Accessed via `fair-esm` package
- Downloads model weights via `esm.pretrained.esm2_t36_3B_UR50D()`

**Authentication:** None required (public model)

**Usage Location:**
- `virnucpro/pipeline/feature_extraction.py`
- Model and alphabet loaded via ESM API

**Configuration:**
- Model variant hardcoded in code
- No external configuration

## Data Storage

### Local Filesystem

**Input:**
- FASTA files (`.fa`, `.fasta`)
- Read via BioPython's SeqIO
- Location: User-specified via CLI `--input` parameter

**Output:**
- CSV files (predictions, consensus)
- Text files (predictions)
- Intermediate `.pt` files (PyTorch tensors)
- Location: User-specified via CLI `--output-dir` parameter

**Intermediate Files:**
- Checkpointed pipeline stages as `.pt` files
- FASTA files for intermediate sequences
- Stored in output directory subdirectories

**No Database:**
- No SQL databases
- No NoSQL databases
- All data in files

## External APIs (None)

**No REST APIs**
**No GraphQL APIs**
**No SOAP Services**

## Authentication & Authorization

**No Authentication Required:**
- HuggingFace models are public
- ESM models are public
- No API keys needed
- No tokens or credentials

## Webhooks & Callbacks

**None Detected**

## Message Queues / Event Systems

**None Detected**

## Monitoring & Observability

### Logging

**Framework:** Python standard library `logging`

**Configuration:**
- `virnucpro/core/logging.py` (76 lines)
- Console-based logging
- No external logging services (Datadog, Sentry, etc.)

**Log Levels:**
- Configurable via logging setup
- No external log aggregation

### Error Tracking

**None Detected:**
- No Sentry
- No Rollbar
- No external error tracking

### Application Performance Monitoring (APM)

**None:**
- No New Relic
- No DataDog APM
- No OpenTelemetry

## CI/CD Integration

**No CI/CD Detected:**
- No GitHub Actions workflows
- No GitLab CI configuration
- No Jenkins files
- No CircleCI config

**Tests Run Locally:**
- Via `pytest` command
- No automated test runs on commit

## Cloud Services

**None Detected:**
- No AWS SDK usage
- No GCP client libraries (beyond Vertex AI for model if used)
- No Azure SDK
- No cloud storage (S3, GCS, Azure Blob)

## Third-Party Services

**None Beyond Model Providers:**
- No payment processors
- No email services (SendGrid, Mailgun)
- No SMS services
- No analytics (Google Analytics, Mixpanel)

## Network Requirements

**Internet Required For:**
1. Initial model downloads:
   - DNABERT-S from HuggingFace Hub
   - ESM-2 from Facebook Research

**After Models Downloaded:**
- Fully offline capable
- All inference runs locally

**Ports:**
- No server ports opened
- No listening services
- Pure CLI application

## Data Flow Summary

```
User FASTA File
    ↓
[virnucpro CLI]
    ↓
Download Models (first run only):
  - HuggingFace Hub → DNABERT-S
  - Facebook ESM → ESM-2 3B
    ↓
[Local Inference Pipeline]
    ↓
Output Files (CSV/TXT)
```

## Dependency on External Services

**Critical:**
- HuggingFace Hub (first run only)
- Facebook Research ESM (first run only)

**Non-Critical:**
- None - once models downloaded, fully standalone

**Failure Modes:**
- Model download failures on first run
- No retry logic detected
- No fallback models
