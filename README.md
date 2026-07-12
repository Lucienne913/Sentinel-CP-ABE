# Sentinel-CP-ABE: Time-Aware Access Control with Threshold Policies for Digital Twins

Academic implementation of **Sentinel-CP-ABE**, a provably secure ciphertext-policy attribute-based encryption framework unifying temporal predicates, THRESHOLD gates, version-based revocation, and AI-driven closed-loop defense for industrial digital twin environments.

## Publication

Transferred to **IEEE Transactions on Information Forensics and Security** (SCI Q1, 2025).

Previous review: Major Revision at **IEEE Internet of Things Journal** (2026).

## Key Features

- **Temporal Access Control**: Hash-chain time token mechanism with forward security; checkpoint-accelerated verifier achieves **O(1) verification (<0.002 ms)** at all chain lengths (10³–10⁵)
- **THRESHOLD(k,n) Policy**: Native threshold gate support (AND, OR, k-n) in a single LSSS representation
- **Version-Based Revocation**: Instant attribute invalidation via version-tag embedding, no re-encryption of existing ciphertexts
- **IND-qCPA-T Security Model**: Quasi-dynamic adversary model bridging selective and adaptive security; tight reduction to DBDH with bound **Adv ≤ 2·Adv_DBDH + (q_H+q_K+q_T)/p**
- **Distributed TA**: (t,n)-threshold secret sharing eliminates single-point key escrow and trust
- **AI Closed-Loop Defense**: Diffusion-based anomaly detection + EWMA adaptive threshold (η=0.1, F1=0.71) + Grad-CAM explainability + dynamic LSSS policy update (mean 4.04 ms)
- **Lightweight Security**: BLS signatures for ciphertext-to-epoch binding; SHA-256 hash chain forward security

## Directory Structure

```
├── src/                      # Core source code
│   ├── t_cp_abe.py           # Sentinel-CP-ABE main implementation
│   ├── setup.py              # Bilinear pairing setup (SS1024 curve)
│   ├── auth.py               # Bidirectional authentication
│   ├── signatures.py         # BLS signatures
│   ├── digital_twin.py       # Digital Twin Manager (Eclipse Ditto)
│   ├── diffusion.py          # Threat diffusion model (DDPM) + EWMA threshold
│   ├── distributed_ta.py     # (t,n)-threshold distributed TA
│   ├── train_diffusion.py    # Model training script
│   ├── subprocess_worker.py  # OOM protection worker
│   └── baselines/            # BSW07, Lightweight ABE, Guo2024, Zhang2024
├── tests/                    # Test suite (23+ files)
│   ├── test_basic.py         # Basic functionality
│   ├── test_security.py      # IND-CPA, EUF-CMA security tests
│   ├── test_performance.py   # Timing benchmarks
│   ├── test_threshold.py     # Threshold policy tests
│   ├── test_dynamic_lsss.py  # AI-driven dynamic LSSS update tests
│   ├── test_hash_chain.py    # Hash chain verification tests
│   ├── test_time_token.py    # Time token tests
│   ├── test_sota_comparison.py # SOTA comparison tests
│   └── ...                   # Additional test files
├── experiments/              # Experiment scripts and results
│   ├── generate_paper_figures.py  # Figure generation
│   ├── preprocess_unsw_nb15.py    # UNSW-NB15 preprocessing
│   ├── evaluate_unsw_nb15.py      # UNSW-NB15 evaluation
│   ├── finetune_unsw_nb15.py      # Diffusion model fine-tuning
│   ├── tune_ewma_eta.py           # EWMA η parameter sweep
│   ├── batch_gradcam_analysis.py  # Grad-CAM batch analysis
│   ├── benchmark_hash_chain.py    # Hash chain O(1) verification benchmark
│   ├── benchmark_dynamic_lsss.py  # Closed-loop latency benchmark
│   ├── benchmark_dta_setup.py     # DTA setup benchmark
│   ├── benchmark_dta_performance.py # DTA performance benchmark
│   ├── test_distributed_ta.py     # Distributed TA integration test
│   ├── test_gradcam.py            # Grad-CAM functionality test
│   └── results/                   # Experiment results (JSON)
├── paper/                    # LaTeX source
│   ├── paper.tex             # Main paper
│   ├── supplementary.tex     # Security proof & extended experiments
│   ├── literature.bib        # Bibliography
│   └── figures/              # System diagrams (drawio + PDF)
├── Dockerfile                # Docker build (Charm-Crypto + PBC)
├── pytest.ini                # pytest configuration
├── requirements.txt          # Python dependencies
└── .gitignore                # Git ignore rules
```

## Installation

### Prerequisites

- Python 3.9+
- Charm-Crypto (requires PBC library)

### Docker (Recommended)

```bash
docker build -t sentinel-cpabe-academic .
docker run -it --rm sentinel-cpabe-academic
```

### Local Installation

```bash
pip install -r requirements.txt
```

See [Dockerfile](Dockerfile) for PBC/Charm-Crypto installation details.

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run by category
python -m pytest tests/ -v -m security
python -m pytest tests/ -v -m performance
python -m pytest tests/ -v -m slow
```

## Test Categories

| Marker | Description |
|--------|-------------|
| `basic` | Basic functionality |
| `security` | IND-qCPA-T, EUF-CMA proofs |
| `performance` | Timing benchmarks |
| `scalability` | Large-scale tests |
| `embedded` | Embedded device simulation |
| `edge_case` | Edge case tests |
| `memory` | Memory leak tests |
| `slow` | Slow tests (>30 seconds) |

## Generate Paper Figures

```bash
python experiments/generate_paper_figures.py
```

Outputs to `paper/figures/`.

## Security Model

- **IND-qCPA-T**: Quasi-dynamic adversary model—selective in time/revocation state, adaptive in LSSS policy
- **DBDH Reduction**: Tight reduction with bound **Adv ≤ 2·Adv_DBDH + (q_H+q_K+q_T)/p**
- **Collusion Resistance**: Proven under threshold-revocation mechanism
- **Forward Security**: Hash-chain time tokens; leaked token cannot decrypt past ciphertexts
- **Distributed Trust**: (t,n)-threshold TA eliminates single-point key escrow

## Performance Highlights

| Operation | Metric | Value |
|-----------|--------|-------|
| KeyGen (10 attrs) | Latency | 159 ms |
| KeyGen (10,000 attrs) | Latency | 167 s |
| Encrypt (all scales) | Latency | 151–171 ms |
| Decrypt (all scales) | Latency | 241–249 ms (CV < 3.5%) |
| Hash Chain Verify (O(1)) | Latency | <0.002 ms |
| Hash Chain Verify (O(N) baseline) | Latency | up to 43 ms |
| Full Closed-Loop | Latency | 60 ms |
| DTA Setup (t=3, n=5) | Latency | 5.25 ms |
| EWMA Detection | F1 (η=0.1) | 0.71 |
| Grad-CAM | Mean active dims | 49.5 ± 2.7 |
| SK Storage | Size | 101 B/attr |
| CT Storage | Size | ~15.6 KB |

## License

Academic research use only.

## Contact

For questions, contact the corresponding author.
