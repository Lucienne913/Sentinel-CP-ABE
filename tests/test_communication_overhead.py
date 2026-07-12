"""
pytest tests/test_communication_overhead.py -m comparison -v

Scheme 4: Digital Twin Network Security Framework - Communication Overhead Tests

Tests communication overhead under different attribute counts:
1. Ciphertext size changes with attribute count
2. Key size changes with attribute count
3. Network transmission time estimation
4. Communication overhead comparison with other schemes

Test content:

- How large are signatures? (BLS signature ~65 bytes)
- How much larger is data after encryption?
Significance for your paper: Proves your system has "high transmission efficiency"

"""

import pytest
import time
import gc
import sys
import json
import pickle
import statistics
from pathlib import Path
from typing import Dict, List, Any
from charm.toolbox.pairinggroup import GT

# Check if pickle supports Charm-Crypto objects
def _safe_pickle_size(obj):
    """Safely compute pickle serialization size, fallback to str if not supported"""
    try:
        return len(pickle.dumps(obj)), True
    except Exception:
        return len(str(obj).encode('utf-8')), False

sys.path.insert(0, str(Path(__file__).parent.parent))

NUM_REPEATS = 100


@pytest.mark.slow
@pytest.mark.comparison
class TestCommunicationOverhead:
    """Communication Overhead Tests"""
    
    ATTR_COUNTS = [10, 50, 100, 500, 1000, 5000]
    NETWORK_SPEEDS = {
        'ethernet': 100,  # Mbps
        'wifi': 50,       # Mbps
        '4g': 20,         # Mbps
        '5g': 1000        # Mbps
    }
    
    @staticmethod
    def _save_overhead_data(data, filename):
        """Save communication overhead data to file (used for SCI paper figures/tables)"""
        output_dir = Path(__file__).parent.parent / 'experiments' / 'results' / 'communication'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def _calculate_transmission_time(size_bytes, network_speed_mbps):
        """Compute network transmission time
        
        Args:
            size_bytes: Data size (bytes)
            network_speed_mbps: Network speed (Mbps)
            
        Returns:
            Transmission time (seconds)
        """
        size_bits = size_bytes * 8
        size_mbits = size_bits / 1_000_000
        return size_mbits / network_speed_mbps
    
    def test_ciphertext_size_overhead(self, system_setup, heartbeat, oom_protection):
        """Ciphertext Size Overhead Test (using binary serialization for accurate measurement, repeated multiple times)"""
        tcabe = system_setup['tcabe']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        NUM_SIZE_REPEATS = 10  # Number of size measurement repeats
        
        results = {}
        total_steps = len(self.ATTR_COUNTS)
        step = 0
        
        for num_attrs in self.ATTR_COUNTS:
            gc.collect()
            
            # Generate attributes and policy
            attrs = [f'attr_{i}' for i in range(num_attrs)]
            policy_str = ' AND '.join(attrs[:min(10, num_attrs)])  # Maximum 10 attributes in policy
            policy_tree = parser.parse(policy_str)
            
            # Repeated measurements
            ct_sizes_pickle = []
            ct_sizes_str = []
            pickle_supported = True
            
            for rep in range(NUM_SIZE_REPEATS):
                M = group.random(GT)
                CT = tcabe.encrypt(M, policy_tree)
                
                # Use pickle serialization to compute accurate size
                size, supported = _safe_pickle_size(CT)
                ct_sizes_pickle.append(size)
                if not supported:
                    pickle_supported = False
                # Also measure str() size for comparison
                ct_sizes_str.append(len(str(CT).encode('utf-8')))
            
            # Compute average value
            ct_size_bytes = statistics.mean(ct_sizes_pickle)
            ct_size_kb = ct_size_bytes / 1024
            ct_str_size_bytes = statistics.mean(ct_sizes_str)
            
            # Compute transmission times under different networks
            transmission_times = {}
            for network, speed in self.NETWORK_SPEEDS.items():
                transmission_times[network] = self._calculate_transmission_time(ct_size_bytes, speed)
            
            results[num_attrs] = {
                'size_bytes': round(ct_size_bytes, 2),
                'size_kb': round(ct_size_kb, 2),
                'size_std_kb': round(statistics.stdev([s/1024 for s in ct_sizes_pickle]), 2) if len(ct_sizes_pickle) > 1 else 0,
                'str_size_bytes': round(ct_str_size_bytes, 2),
                'compression_ratio': round(ct_size_bytes / ct_str_size_bytes, 4) if ct_str_size_bytes > 0 else 0,
                'transmission_times': transmission_times,
                'num_repeats': NUM_SIZE_REPEATS
            }
            
            step += 1
            heartbeat(step, total_steps, f'CT Size: {num_attrs} attrs → {ct_size_kb:.2f} KB (pickle), {ct_str_size_bytes/1024:.2f} KB (str)')
            oom_protection.check_and_protect()
        
        self._save_overhead_data(results, 'ciphertext_size_overhead.json')
        
        print("\n" + "=" * 80)
        print(f"Ciphertext Size Overhead ({NUM_SIZE_REPEATS} repeated measurements, pickle binary serialization)")
        print("=" * 80)
        print(f"{'AttributeCount':12} {'pickle(KB)':12} {'std(KB)':10} {'str(KB)':12} {'Compression':10} {'Ethernet(ms)':12} {'4G(ms)':12} {'5G(ms)':12}")
        print("-" * 80)
        for num_attrs, data in results.items():
            times = data['transmission_times']
            print(f"{num_attrs:12} {data['size_kb']:11.2f} {data['size_std_kb']:9.2f} {data['str_size_bytes']/1024:11.2f} {data['compression_ratio']:9.2%} " +
                  f"{times['ethernet']*1000:10.2f} " +
                  f"{times['4g']*1000:10.2f} " +
                  f"{times['5g']*1000:10.2f}")
    
    def test_key_size_overhead(self, system_setup, heartbeat, oom_protection):
        """Key Size Overhead Test (using binary serialization for accurate measurement, repeated multiple times)"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        
        NUM_SIZE_REPEATS = 10  # Number of size measurement repeats
        
        results = {}
        total_steps = len(self.ATTR_COUNTS)
        step = 0
        
        for num_attrs in self.ATTR_COUNTS:
            gc.collect()
            
            # Repeated measurements
            sk_sizes_pickle = []
            sk_sizes_str = []
            pickle_supported = True
            
            for rep in range(NUM_SIZE_REPEATS):
                # Generate key
                attrs = [f'attr_{i}' for i in range(num_attrs)]
                SK = tcabe.keygen(MSK, attrs)
                
                # Use pickle serialization to compute accurate size
                size, supported = _safe_pickle_size(SK)
                sk_sizes_pickle.append(size)
                if not supported:
                    pickle_supported = False
                # Also measure str() size for comparison
                sk_sizes_str.append(len(str(SK).encode('utf-8')))
            
            # Compute average value
            sk_size_bytes = statistics.mean(sk_sizes_pickle)
            sk_size_kb = sk_size_bytes / 1024
            sk_str_size_bytes = statistics.mean(sk_sizes_str)
            
            # Compute transmission times under different networks
            transmission_times = {}
            for network, speed in self.NETWORK_SPEEDS.items():
                transmission_times[network] = self._calculate_transmission_time(sk_size_bytes, speed)
            
            results[num_attrs] = {
                'size_bytes': round(sk_size_bytes, 2),
                'size_kb': round(sk_size_kb, 2),
                'size_std_kb': round(statistics.stdev([s/1024 for s in sk_sizes_pickle]), 2) if len(sk_sizes_pickle) > 1 else 0,
                'str_size_bytes': round(sk_str_size_bytes, 2),
                'compression_ratio': round(sk_size_bytes / sk_str_size_bytes, 4) if sk_str_size_bytes > 0 else 0,
                'transmission_times': transmission_times,
                'num_repeats': NUM_SIZE_REPEATS
            }
            
            step += 1
            heartbeat(step, total_steps, f'Key Size: {num_attrs} attrs → {sk_size_kb:.2f} KB (pickle), {sk_str_size_bytes/1024:.2f} KB (str)')
            oom_protection.check_and_protect()
        
        self._save_overhead_data(results, 'key_size_overhead.json')
        
        print("\n" + "=" * 80)
        print(f"Key Size Overhead ({NUM_SIZE_REPEATS} repeated measurements, pickle binary serialization)")
        print("=" * 80)
        print(f"{'AttributeCount':12} {'pickle(KB)':12} {'std(KB)':10} {'str(KB)':12} {'Compression':10} {'Ethernet(ms)':12} {'4G(ms)':12} {'5G(ms)':12}")
        print("-" * 80)
        for num_attrs, data in results.items():
            times = data['transmission_times']
            print(f"{num_attrs:12} {data['size_kb']:11.2f} {data['size_std_kb']:9.2f} {data['str_size_bytes']/1024:11.2f} {data['compression_ratio']:9.2%} " +
                  f"{times['ethernet']*1000:10.2f} " +
                  f"{times['4g']*1000:10.2f} " +
                  f"{times['5g']*1000:10.2f}")
    
    def test_communication_overhead_comparison(self, system_setup, heartbeat, oom_protection):
        """Communication Overhead Comparison Test (100 repeats, using pickle binary serialization)"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']

        attr_counts = [10, 100, 500]
        results = {}
        total_steps = len(attr_counts)
        step = 0

        for num_attrs in attr_counts:
            gc.collect()
            attrs = [f'attr_{i}' for i in range(num_attrs)]
            policy_str = ' AND '.join(attrs[:min(10, num_attrs)])
            policy_tree = parser.parse(policy_str)

            ct_sizes_pickle = []
            ct_sizes_str = []
            sk_sizes_pickle = []
            sk_sizes_str = []
            encrypt_times = []
            keygen_times = []

            for rep in range(NUM_REPEATS):
                M = group.random(GT)

                start = time.time()
                CT = tcabe.encrypt(M, policy_tree)
                encrypt_times.append((time.time() - start) * 1000)
                ct_size, _ = _safe_pickle_size(CT)
                ct_sizes_pickle.append(ct_size)
                ct_sizes_str.append(len(str(CT).encode('utf-8')))

                start = time.time()
                SK = tcabe.keygen(MSK, attrs)
                keygen_times.append((time.time() - start) * 1000)
                sk_size, _ = _safe_pickle_size(SK)
                sk_sizes_pickle.append(sk_size)
                sk_sizes_str.append(len(str(SK).encode('utf-8')))

            ct_size_kb = statistics.mean(ct_sizes_pickle) / 1024
            sk_size_kb = statistics.mean(sk_sizes_pickle) / 1024
            ct_size_str_kb = statistics.mean(ct_sizes_str) / 1024
            sk_size_str_kb = statistics.mean(sk_sizes_str) / 1024

            transmission_times = {}
            for network, speed in self.NETWORK_SPEEDS.items():
                transmission_times[network] = self._calculate_transmission_time(ct_size_kb * 1024, speed)

            results[num_attrs] = {
                'ct_size_kb': round(ct_size_kb, 2),
                'sk_size_kb': round(sk_size_kb, 2),
                'ct_size_str_kb': round(ct_size_str_kb, 2),
                'sk_size_str_kb': round(sk_size_str_kb, 2),
                'ct_compression_ratio': round(ct_size_kb / ct_size_str_kb, 4) if ct_size_str_kb > 0 else 0,
                'sk_compression_ratio': round(sk_size_kb / sk_size_str_kb, 4) if sk_size_str_kb > 0 else 0,
                'ct_size_bytes_mean': round(statistics.mean(ct_sizes_pickle), 2),
                'sk_size_bytes_mean': round(statistics.mean(sk_sizes_pickle), 2),
                'encrypt_mean_ms': round(statistics.mean(encrypt_times), 2),
                'encrypt_std_ms': round(statistics.stdev(encrypt_times), 2),
                'keygen_mean_ms': round(statistics.mean(keygen_times), 2),
                'keygen_std_ms': round(statistics.stdev(keygen_times), 2),
                'transmission_times': {k: round(v * 1000, 2) for k, v in transmission_times.items()},
                'num_repeats': NUM_REPEATS,
            }

            step += 1
            heartbeat(step, total_steps,
                      f'OurScheme {num_attrs} attrs ({NUM_REPEATS} reps): CT={ct_size_kb:.2f}KB (pickle), {ct_size_str_kb:.2f}KB (str)')
            oom_protection.check_and_protect()

        self._save_overhead_data(results, 'communication_overhead_comparison.json')

        print("\n" + "=" * 120)
        print(f"Our Scheme Communication Overhead ({NUM_REPEATS} repeated measurements, pickle binary serialization)")
        print("=" * 120)
        print(f"{'AttrCount':10} {'CT(pickle)':12} {'CT(str)':12} {'Compression':10} {'SK(pickle)':12} {'SK(str)':12} {'Compression':10} {'Enc mean':12} {'4G(ms)':12} {'5G(ms)':12}")
        print("-" * 120)
        for num_attrs, data in results.items():
            t = data['transmission_times']
            print(f"{num_attrs:10} {data['ct_size_kb']:11.2f} {data['ct_size_str_kb']:11.2f} {data['ct_compression_ratio']:9.2%} "
                  f"{data['sk_size_kb']:11.2f} {data['sk_size_str_kb']:11.2f} {data['sk_compression_ratio']:9.2%} "
                  f"{data['encrypt_mean_ms']:10.2f}ms "
                  f"{t['ethernet']:10.2f} {t['5g']:10.2f}")
        print("=" * 120)
    
    def test_generate_communication_overhead_table(self, system_setup, heartbeat, oom_protection):
        """Generate Communication Overhead Table (used for SCI paper Table 3) - using pickle binary serialization for accurate measurement"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']

        results = {}
        total_steps = len(self.ATTR_COUNTS)
        step = 0

        for num_attrs in self.ATTR_COUNTS:
            gc.collect()
            attrs = [f'attr_{i}' for i in range(num_attrs)]
            policy_str = ' AND '.join(attrs[:min(10, num_attrs)])
            policy_tree = parser.parse(policy_str)
            M = group.random(GT)

            CT = tcabe.encrypt(M, policy_tree)
            # Use pickle serialization to compute accurate size
            ct_size, _ = _safe_pickle_size(CT)
            ct_size_kb = ct_size / 1024
            ct_size_str_kb = len(str(CT).encode('utf-8')) / 1024

            SK = tcabe.keygen(MSK, attrs)
            # Use pickle serialization to compute accurate size
            sk_size, _ = _safe_pickle_size(SK)
            sk_size_kb = sk_size / 1024
            sk_size_str_kb = len(str(SK).encode('utf-8')) / 1024

            time_4g = self._calculate_transmission_time((ct_size_kb + sk_size_kb) * 1024, 20) * 1000
            time_5g = self._calculate_transmission_time((ct_size_kb + sk_size_kb) * 1024, 1000) * 1000

            results[num_attrs] = {
                'ct_size_kb': round(ct_size_kb, 2),
                'sk_size_kb': round(sk_size_kb, 2),
                'ct_size_str_kb': round(ct_size_str_kb, 2),
                'sk_size_str_kb': round(sk_size_str_kb, 2),
                'ct_compression_ratio': round(ct_size_kb / ct_size_str_kb, 4) if ct_size_str_kb > 0 else 0,
                'sk_compression_ratio': round(sk_size_kb / sk_size_str_kb, 4) if sk_size_str_kb > 0 else 0,
                'time_4g_ms': round(time_4g, 2),
                'time_5g_ms': round(time_5g, 2)
            }

            step += 1
            heartbeat(step, total_steps, f'Overhead: {num_attrs} attrs → CT={ct_size_kb:.2f}KB (pickle), SK={sk_size_kb:.2f}KB (pickle)')
            oom_protection.check_and_protect()

        self._save_overhead_data(results, 'communication_overhead_table.json')

        print("\n" + "=" * 120)
        print("Table 3: Communication Overhead Test Results (using pickle binary serialization for accurate measurement)")
        print("=" * 120)
        print(f"{'AttributeCount':12} {'CT(pickle/KB)':15} {'CT(str/KB)':15} {'Compression':10} {'SK(pickle/KB)':15} {'SK(str/KB)':15} {'Compression':10} {'4G(ms)':12} {'5G(ms)':12}")
        print("-" * 120)
        for num_attrs in self.ATTR_COUNTS:
            r = results[num_attrs]
            print(f"{num_attrs:12} {r['ct_size_kb']:14.2f} {r['ct_size_str_kb']:14.2f} {r['ct_compression_ratio']:9.2%} "
                  f"{r['sk_size_kb']:14.2f} {r['sk_size_str_kb']:14.2f} {r['sk_compression_ratio']:9.2%} "
                  f"{r['time_4g_ms']:11.2f} {r['time_5g_ms']:11.2f}")
        print("=" * 120)
