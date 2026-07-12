#!/usr/bin/env python3
"""
Embedded Device Simulation Test Script
Embedded Device Concurrent Simulation Tests

Simulate multi-device concurrent scenarios using real cryptographic operations (Charm-Crypto):
- Multi-thread simulation of multiple IoT devices simultaneously initiating authentication/encryption requests
- Real keygen/encrypt/decrypt operations
- Real latency and memory measurements
"""

import pytest
import time
import gc
import sys
import os
import json
import threading
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from charm.toolbox.pairinggroup import GT

sys.path.insert(0, str(Path(__file__).parent.parent))


class CryptoDeviceWorker:
    """Single device worker thread, executes real cryptographic operations."""

    def __init__(self, device_id: str, tcabe, MSK, PP, group, parser,
                 num_attrs: int = 10, num_requests: int = 5):
        self.device_id = device_id
        self.tcabe = tcabe
        self.MSK = MSK
        self.PP = PP
        self.group = group
        self.parser = parser
        self.num_attrs = num_attrs
        self.num_requests = num_requests
        self.results = []
        self._lock = threading.Lock()

    def run(self):
        """Execute real cryptographic operation sequence."""
        attrs = [f'attr_{i}' for i in range(self.num_attrs)]
        policy_str = ' AND '.join(attrs[:min(5, self.num_attrs)])
        policy_tree = self.parser.parse(policy_str)

        process = psutil.Process(os.getpid())

        for i in range(self.num_requests):
            try:
                mem_before = process.memory_info().rss / (1024 * 1024)

                start = time.time()
                SK = self.tcabe.keygen(self.MSK, attrs)
                keygen_time = time.time() - start

                M = self.group.random(GT)

                start = time.time()
                CT = self.tcabe.encrypt(M, policy_tree)
                encrypt_time = time.time() - start

                start = time.time()
                decrypted = self.tcabe.decrypt(SK, CT)
                decrypt_time = time.time() - start

                mem_after = process.memory_info().rss / (1024 * 1024)

                success = (decrypted == M)

                with self._lock:
                    self.results.append({
                        'device_id': self.device_id,
                        'request_idx': i,
                        'success': success,
                        'keygen_ms': round(keygen_time * 1000, 2),
                        'encrypt_ms': round(encrypt_time * 1000, 2),
                        'decrypt_ms': round(decrypt_time * 1000, 2),
                        'total_ms': round((keygen_time + encrypt_time + decrypt_time) * 1000, 2),
                        'memory_before_mb': round(mem_before, 2),
                        'memory_after_mb': round(mem_after, 2),
                    })
            except Exception as e:
                with self._lock:
                    self.results.append({
                        'device_id': self.device_id,
                        'request_idx': i,
                        'success': False,
                        'error': str(e),
                        'total_ms': 0,
                    })

    def get_stats(self) -> Dict:
        """Get statistics for this device."""
        with self._lock:
            if not self.results:
                return {'device_id': self.device_id, 'total_requests': 0}

            successful = [r for r in self.results if r.get('success')]
            all_times = [r['total_ms'] for r in self.results if r.get('total_ms', 0) > 0]

            return {
                'device_id': self.device_id,
                'num_attrs': self.num_attrs,
                'total_requests': len(self.results),
                'successful_requests': len(successful),
                'success_rate': round(len(successful) / len(self.results), 4) if self.results else 0,
                'avg_latency_ms': round(sum(all_times) / len(all_times), 2) if all_times else 0,
                'min_latency_ms': round(min(all_times), 2) if all_times else 0,
                'max_latency_ms': round(max(all_times), 2) if all_times else 0,
            }


@pytest.mark.slow
@pytest.mark.embedded
class TestEmbeddedDeviceSimulation:
    """Embedded Device Concurrent Simulation Tests (real cryptographic operations)"""

    @staticmethod
    def _save_results(data, filename):
        output_dir = Path(__file__).parent.parent / 'experiments' / 'results' / 'embedded'
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def test_concurrent_device_simulation(self, system_setup, heartbeat, oom_protection):
        """Multi-device concurrent cryptographic operation simulation"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']

        device_configs = [
            {'id': 'sensor_001', 'attrs': 5, 'requests': 3},
            {'id': 'sensor_002', 'attrs': 10, 'requests': 3},
            {'id': 'sensor_003', 'attrs': 20, 'requests': 3},
            {'id': 'actuator_001', 'attrs': 5, 'requests': 3},
            {'id': 'actuator_002', 'attrs': 10, 'requests': 3},
            {'id': 'gateway_001', 'attrs': 15, 'requests': 3},
            {'id': 'gateway_002', 'attrs': 20, 'requests': 3},
            {'id': 'controller_001', 'attrs': 10, 'requests': 3},
            {'id': 'controller_002', 'attrs': 15, 'requests': 3},
            {'id': 'edge_001', 'attrs': 5, 'requests': 3},
        ]

        workers = []
        for cfg in device_configs:
            worker = CryptoDeviceWorker(
                device_id=cfg['id'], tcabe=tcabe, MSK=MSK, PP=PP,
                group=group, parser=parser,
                num_attrs=cfg['attrs'], num_requests=cfg['requests']
            )
            workers.append(worker)

        gc.collect()
        mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

        start = time.time()
        threads = []
        for worker in workers:
            t = threading.Thread(target=worker.run, daemon=True)
            threads.append(t)
            t.start()

        total_steps = len(threads)
        for i, t in enumerate(threads):
            t.join()
            heartbeat(i + 1, total_steps, f'Device {device_configs[i]["id"]} completed')

        wall_time = time.time() - start
        mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        oom_protection.check_and_protect()

        all_stats = [w.get_stats() for w in workers]
        total_requests = sum(s['total_requests'] for s in all_stats)
        total_success = sum(s['successful_requests'] for s in all_stats)
        all_latencies = [s['avg_latency_ms'] for s in all_stats if s['avg_latency_ms'] > 0]

        summary = {
            'timestamp': datetime.now().isoformat(),
            'num_devices': len(device_configs),
            'total_requests': total_requests,
            'successful_requests': total_success,
            'success_rate': round(total_success / total_requests, 4) if total_requests > 0 else 0,
            'wall_time_s': round(wall_time, 2),
            'overall_avg_latency_ms': round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else 0,
            'memory_before_mb': round(mem_before, 2),
            'memory_after_mb': round(mem_after, 2),
            'memory_delta_mb': round(mem_after - mem_before, 2),
        }

        results = {'summary': summary, 'devices': all_stats}
        self._save_results(results, 'concurrent_device_simulation.json')

        print("\n" + "=" * 90)
        print("Multi-device Concurrent Simulation Results (real cryptographic operations)")
        print("=" * 90)
        print(f"{'DeviceID':18} {'AttrCount':8} {'Requests':8} {'SuccessRate':10} {'AvgLatency(ms)':14} {'Min(ms)':10} {'Max(ms)':10}")
        print("-" * 90)
        for s in all_stats:
            print(f"{s['device_id']:18} {s.get('num_attrs', '-'):>6} {s['total_requests']:>7} "
                  f"{s['success_rate']*100:>8.1f}% {s['avg_latency_ms']:>12.1f} "
                  f"{s.get('min_latency_ms', 0):>9.1f} {s.get('max_latency_ms', 0):>9.1f}")
        print("-" * 90)
        print(f"Total: {total_requests} requests, {total_success} Success, "
              f"SuccessRate {summary['success_rate']*100:.1f}%, "
              f"TotalTime {summary['wall_time_s']:.1f}s, "
              f"MemoryDelta {summary['memory_delta_mb']:.1f}MB")
        print("=" * 90)

    def test_scalable_concurrent_simulation(self, system_setup, heartbeat, oom_protection):
        """Scalable concurrent test - real throughput under different device counts"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']

        device_counts = [1, 3, 5, 10]
        results = {}
        total_steps = len(device_counts)
        step = 0

        for num_devices in device_counts:
            gc.collect()
            workers = []
            for i in range(num_devices):
                worker = CryptoDeviceWorker(
                    device_id=f'device_{i:03d}', tcabe=tcabe, MSK=MSK, PP=PP,
                    group=group, parser=parser, num_attrs=10, num_requests=2
                )
                workers.append(worker)

            start = time.time()
            threads = []
            for worker in workers:
                t = threading.Thread(target=worker.run, daemon=True)
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
            wall_time = time.time() - start

            all_stats = [w.get_stats() for w in workers]
            total_requests = sum(s['total_requests'] for s in all_stats)
            total_success = sum(s['successful_requests'] for s in all_stats)
            all_latencies = [s['avg_latency_ms'] for s in all_stats if s['avg_latency_ms'] > 0]

            results[num_devices] = {
                'num_devices': num_devices,
                'total_requests': total_requests,
                'success_rate': round(total_success / total_requests, 4) if total_requests > 0 else 0,
                'wall_time_s': round(wall_time, 2),
                'throughput_ops': round(total_requests / wall_time, 2) if wall_time > 0 else 0,
                'avg_latency_ms': round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else 0,
            }

            step += 1
            heartbeat(step, total_steps,
                      f'{num_devices} devices: {total_requests} req in {wall_time:.1f}s, '
                      f'throughput={results[num_devices]["throughput_ops"]:.1f} ops/s')
            oom_protection.check_and_protect()

        self._save_results(results, 'scalable_concurrent_simulation.json')

        print("\n" + "=" * 80)
        print("Scalable Concurrent Test Results (real cryptographic operations)")
        print("=" * 80)
        print(f"{'Devices':10} {'Requests':10} {'SuccessRate':10} {'TotalTime(s)':12} {'Throughput(op/s)':14} {'AvgLatency(ms)':14}")
        print("-" * 80)
        for num_devices, r in results.items():
            print(f"{num_devices:10} {r['total_requests']:10} {r['success_rate']*100:>8.1f}% "
                  f"{r['wall_time_s']:10.2f} {r['throughput_ops']:12.2f} {r['avg_latency_ms']:12.2f}")
        print("=" * 80)
