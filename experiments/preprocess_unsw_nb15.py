#!/usr/bin/env python3
"""
UNSW-NB15 Dataset Preprocessing for T-CP-ABE Evaluation

Maps real network traffic features to ABE attribute space.
Downloads and processes the actual UNSW-NB15 dataset.

Usage:
    python preprocess_unsw_nb15.py [--data-dir ./data/unsw_nb15] [--max-records 5000]
"""

import json
import hashlib
import time
import argparse
import urllib.request
import os
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

UNSW_NB15_CSV_URLS = [
    "https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files=UNSW-NB15_1.csv",
    "https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files=UNSW-NB15_2.csv",
    "https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files=UNSW-NB15_3.csv",
    "https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files=UNSW-NB15_4.csv",
]

FEATURE_NAMES = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
    'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
    'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb',
    'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
    'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt',
    'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
    'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'attack_cat', 'label'
]

PROTO_CATEGORIES = ['tcp', 'udp', 'arp', 'ospf', 'icmp', 'igmp', 'rtp',
                    'udt', 'sctp', 'gre', 'esp', 'isis']

SERVICE_CATEGORIES = ['-', 'http', 'ftp', 'smtp', 'ssh', 'dns', 'ftp-data',
                      'pop3', 'snmp', 'ssl', 'dhcp', 'irc', 'radius', 'pop',
                      'imap', 'ident', 'ntp', 'ircu']

STATE_CATEGORIES = ['INT', 'CON', 'FIN', 'REQ', 'RST', 'ACC', 'CLO', 'ECO',
                    'PAR', 'URN', 'no']

ATTACK_CATEGORIES = [
    'Normal', 'Fuzzers', 'Analysis', 'Backdoors', 'DoS', 'Exploits',
    'Generic', 'Reconnaissance', 'Shellcode', 'Worms'
]

THREAT_LEVEL_MAP = {
    'normal': 'threat:none',
    'fuzzers': 'threat:low',
    'analysis': 'threat:low',
    'reconnaissance': 'threat:low',
    'dos': 'threat:medium',
    'exploits': 'threat:medium',
    'generic': 'threat:medium',
    'backdoors': 'threat:high',
    'backdoor': 'threat:high',
    'shellcode': 'threat:high',
    'worms': 'threat:critical',
}

TRAFFIC_VOLUME_BINS = {
    'low': (0, 1000),
    'medium': (1000, 100000),
    'high': (100000, float('inf')),
}

OUTPUT_DIR = Path(__file__).parent / 'results' / 'unsw_nb15'


def download_dataset(data_dir: Path) -> List[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_files = []

    for i, url in enumerate(UNSW_NB15_CSV_URLS, 1):
        filename = f'UNSW-NB15_{i}.csv'
        filepath = data_dir / filename
        if filepath.exists():
            print(f"  [skip] {filename} already exists")
            csv_files.append(filepath)
            continue
        print(f"  [download] {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            csv_files.append(filepath)
            print(f"  [done] {filename} ({filepath.stat().st_size / 1024 / 1024:.1f} MB)")
        except Exception as e:
            print(f"  [warn] Failed to download {filename}: {e}")
            print(f"  Please download manually from: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
            print(f"  Place CSV files in: {data_dir}")
    return csv_files


def load_csv_data(csv_files: List[Path], max_records: int = 0) -> List[Dict]:
    records = []
    for csv_file in csv_files:
        print(f"  Loading {csv_file.name}...")
        with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                continue
            header = [h.strip().lower() for h in header]
            for line_num, row in enumerate(reader, 1):
                if len(row) < len(header):
                    row.extend([''] * (len(header) - len(row)))
                record = {}
                for idx, h in enumerate(header):
                    if idx < len(row):
                        record[h] = row[idx].strip()
                    else:
                        record[h] = ''
                records.append(record)
                if max_records > 0 and len(records) >= max_records:
                    break
            if max_records > 0 and len(records) >= max_records:
                break
    print(f"  Loaded {len(records)} records total")
    return records


def record_to_abe_attributes(record: Dict) -> List[str]:
    attrs = []

    proto = record.get('proto', 'tcp').lower().strip()
    if proto in PROTO_CATEGORIES:
        attrs.append(f'protocol:{proto}')
    else:
        attrs.append('protocol:other')

    service = record.get('service', '-').lower().strip()
    if service in SERVICE_CATEGORIES:
        attrs.append(f'service:{service}')
    else:
        attrs.append('service:other')

    state = record.get('state', 'INT').upper().strip()
    if state in STATE_CATEGORIES:
        attrs.append(f'state:{state}')
    else:
        attrs.append('state:other')

    try:
        total_bytes = int(record.get('sbytes', 0)) + int(record.get('dbytes', 0))
    except (ValueError, TypeError):
        total_bytes = 0
    for level, (lo, hi) in TRAFFIC_VOLUME_BINS.items():
        if lo <= total_bytes < hi:
            attrs.append(f'volume:{level}')
            break

    attack_cat = record.get('attack_label', record.get('attack_cat', 'normal')).strip().lower()
    if not attack_cat or attack_cat == '':
        attack_cat = 'normal'
    threat = THREAT_LEVEL_MAP.get(attack_cat, 'threat:low')
    attrs.append(threat)

    try:
        dur = float(record.get('dur', 0))
    except (ValueError, TypeError):
        dur = 0
    if dur < 1:
        attrs.append('duration:short')
    elif dur < 60:
        attrs.append('duration:medium')
    else:
        attrs.append('duration:long')

    return attrs


def attributes_to_policy(attrs: List[str], operator: str = 'AND') -> str:
    return '(' + f' {operator} '.join(attrs) + ')'


def compute_attr_hash(attr: str) -> str:
    return hashlib.sha256(attr.encode()).hexdigest()[:16]


def run_preprocessing(data_dir: Path, max_records: int = 5000, force: bool = False):
    print("=" * 80)
    print("UNSW-NB15 Dataset Preprocessing for T-CP-ABE")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Offline cache check: if preprocessing results already exist, skip download and processing
    summary_file = OUTPUT_DIR / 'unsw_nb15_summary.json'
    stats_file = OUTPUT_DIR / 'preprocessing_stats.json'
    if not force and summary_file.exists() and stats_file.exists():
        print("\n[CACHE HIT] Pre-processed results already exist:")
        print(f"  Summary:    {summary_file}")
        print(f"  Stats:      {stats_file}")
        print("  Skipping download and processing. Use --force to re-run.")
        print("=" * 80)
        with open(stats_file, 'r', encoding='utf-8') as f:
            cached_stats = json.load(f)
        print(f"  Cached data: {cached_stats.get('total_records', 'N/A')} records, "
              f"{cached_stats.get('unique_attributes', 'N/A')} unique attributes")
        return cached_stats

    print("\n[Step 1] Download dataset")
    csv_files = download_dataset(data_dir)
    if not csv_files:
        print("ERROR: No CSV files available. Exiting.")
        return None

    print("\n[Step 2] Load CSV data")
    records = load_csv_data(csv_files, max_records)
    if not records:
        print("ERROR: No records loaded. Exiting.")
        return None

    print("\n[Step 3] Map to ABE attributes")
    all_attrs = set()
    attr_records = []
    attack_dist = {}

    for record in records:
        attrs = record_to_abe_attributes(record)
        attr_records.append(attrs)
        all_attrs.update(attrs)
        attack_cat = record.get('attack_label', record.get('attack_cat', 'normal')).strip().lower()
        if not attack_cat:
            attack_cat = 'normal'
        attack_dist[attack_cat] = attack_dist.get(attack_cat, 0) + 1

    print(f"  Unique attributes: {len(all_attrs)}")
    print(f"  Attack distribution:")
    for cat, count in sorted(attack_dist.items(), key=lambda x: -x[1]):
        print(f"    {cat:20s}: {count:6d} ({count/len(records)*100:.1f}%)")

    print("\n[Step 4] Generate attribute statistics")
    attr_freq = {}
    for attrs in attr_records:
        for attr in attrs:
            attr_freq[attr] = attr_freq.get(attr, 0) + 1

    attr_stats = {
        'timestamp': datetime.now().isoformat(),
        'data_source': 'real_UNSW-NB15',
        'total_records': len(records),
        'unique_attributes': len(all_attrs),
        'attack_distribution': attack_dist,
        'attribute_frequency': dict(sorted(attr_freq.items(), key=lambda x: -x[1])[:30]),
        'attribute_categories': {
            'protocol': len([a for a in all_attrs if a.startswith('protocol:')]),
            'service': len([a for a in all_attrs if a.startswith('service:')]),
            'state': len([a for a in all_attrs if a.startswith('state:')]),
            'volume': len([a for a in all_attrs if a.startswith('volume:')]),
            'threat': len([a for a in all_attrs if a.startswith('threat:')]),
            'duration': len([a for a in all_attrs if a.startswith('duration:')]),
        }
    }

    with open(OUTPUT_DIR / 'preprocessing_stats.json', 'w', encoding='utf-8') as f:
        json.dump(attr_stats, f, indent=2, ensure_ascii=False)

    print("\n[Step 5] Run T-CP-ABE timing experiments")
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, GT, pair
        from src.setup import T_CP_ABE_Setup
        from src.t_cp_abe import T_CP_ABE, PolicyParser

        setup = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
        PP, MSK = setup.setup()
        tcabe = T_CP_ABE(PP)
        group = PP['group']
        parser = PolicyParser()

        attr_counts = [3, 5, 7, 10, 15]
        timing_results = {}

        for n_attrs in attr_counts:
            sample_records = attr_records[:min(200, len(attr_records))]
            keygen_times = []
            encrypt_times = []
            decrypt_times = []
            success_count = 0

            for attrs in sample_records[:100]:
                selected_attrs = attrs[:n_attrs]
                if len(selected_attrs) < 2:
                    continue

                policy_str = attributes_to_policy(selected_attrs)
                try:
                    policy_tree = parser.parse(policy_str)
                except Exception:
                    continue

                try:
                    start = time.time()
                    SK = tcabe.keygen(MSK, selected_attrs)
                    keygen_times.append(time.time() - start)

                    M = group.random(GT)
                    start = time.time()
                    CT = tcabe.encrypt(M, policy_tree)
                    encrypt_times.append(time.time() - start)

                    start = time.time()
                    decrypted = tcabe.decrypt(SK, CT)
                    decrypt_times.append(time.time() - start)

                    if decrypted == M:
                        success_count += 1
                except Exception:
                    continue

            if keygen_times:
                timing_results[f'attrs_{n_attrs}'] = {
                    'num_attributes': n_attrs,
                    'num_samples': len(keygen_times),
                    'success_count': success_count,
                    'keygen_mean_ms': round(sum(keygen_times) / len(keygen_times) * 1000, 3),
                    'encrypt_mean_ms': round(sum(encrypt_times) / len(encrypt_times) * 1000, 3),
                    'decrypt_mean_ms': round(sum(decrypt_times) / len(decrypt_times) * 1000, 3),
                    'keygen_p95_ms': round(sorted(keygen_times)[int(len(keygen_times) * 0.95)] * 1000, 3),
                    'encrypt_p95_ms': round(sorted(encrypt_times)[int(len(encrypt_times) * 0.95)] * 1000, 3),
                    'decrypt_p95_ms': round(sorted(decrypt_times)[int(len(decrypt_times) * 0.95)] * 1000, 3),
                    'decrypt_success_rate': round(success_count / len(keygen_times) * 100, 1),
                }
                print(f"  {n_attrs:2d} attrs: keygen={timing_results[f'attrs_{n_attrs}']['keygen_mean_ms']:.2f}ms, "
                      f"encrypt={timing_results[f'attrs_{n_attrs}']['encrypt_mean_ms']:.2f}ms, "
                      f"decrypt={timing_results[f'attrs_{n_attrs}']['decrypt_mean_ms']:.2f}ms, "
                      f"success={success_count}/{len(keygen_times)}")

        with open(OUTPUT_DIR / 'unsw_nb15_timing_results.json', 'w', encoding='utf-8') as f:
            json.dump(timing_results, f, indent=2)

    except ImportError as e:
        print(f"  [skip] Charm-Crypto not available: {e}")
        print("  Run this script inside the Docker container for timing experiments.")

    print("\n[Step 6] Generate summary report")
    summary = {
        'dataset': 'UNSW-NB15',
        'paper_section': 'Section VII: Real-World Validation',
        'description': 'Network traffic features mapped to ABE attribute space',
        'data_source': 'real_UNSW-NB15_dataset',
        'total_records_processed': len(records),
        'unique_abe_attributes': len(all_attrs),
        'attack_categories': len(attack_dist),
        'timing_experiments': 'unsw_nb15_timing_results.json',
        'generated_at': datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / 'unsw_nb15_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)

    return attr_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UNSW-NB15 Preprocessing for T-CP-ABE')
    parser.add_argument('--data-dir', type=str, default='./data/unsw_nb15',
                        help='Directory to store/download UNSW-NB15 CSV files')
    parser.add_argument('--max-records', type=int, default=5000,
                        help='Maximum records to process (0 = all)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if cached results exist')
    args = parser.parse_args()

    run_preprocessing(Path(args.data_dir), args.max_records, force=args.force)
