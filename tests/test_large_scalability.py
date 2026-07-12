"""
Scheme 4: Digital Twin Network Security Framework - Large Scale Scalability Tests

Test scalability from 10 to 100000+ scale:
1. Key Generation scalability
2. Encrypt/Decrypt scalability
3. Policy Tree Complexity scalability
4. Digital Twin management scalability
"""

import pytest
import time
import gc
import sys
import statistics
from pathlib import Path
from typing import Dict, List, Any
from charm.toolbox.pairinggroup import GT

sys.path.insert(0, str(Path(__file__).parent.parent))

NUM_REPEATS = 100


def get_repeats(num_attrs):
    """Adaptive repeat count: 100 repeats for small scale, reduced for large scale to ensure feasibility"""
    if num_attrs <= 500:
        return 100
    elif num_attrs <= 1000:
        return 20
    elif num_attrs <= 5000:
        return 10
    else:
        return 5


@pytest.mark.slow
@pytest.mark.comparison
class TestLargeScalability:
    """Large scale scalability tests (10 to 100000 scale)"""
    
    SCALES = [10, 50, 100, 500, 1000, 5000, 10000]
    
    @staticmethod
    def _save_scalability_data(data, filename):
        """Save scalability data to file (used for SCI paper figures)"""
        import json
        output_dir = Path(__file__).parent.parent / 'experiments' / 'results' / 'scalability'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def _generate_simple_policy(num_attrs):
        """Generate simple policy (AND joined)"""
        attrs = [f'attr_{i}' for i in range(min(10, num_attrs))]
        return ' AND '.join(attrs)
    
    @staticmethod
    def _generate_complex_policy(num_attrs):
        """Generate complex policy (nested AND/OR)"""
        attrs = [f'attr_{i}' for i in range(min(20, num_attrs))]
        if len(attrs) < 3:
            return ' AND '.join(attrs)
        elif len(attrs) < 6:
            return f'( {attrs[0]} AND {attrs[1]} ) OR {attrs[2]}'
        elif len(attrs) < 10:
            return f'( ( {attrs[0]} AND {attrs[1]} ) OR {attrs[2]} ) AND {attrs[3]}'
        else:
            return f'( ( ( {attrs[0]} AND {attrs[1]} ) OR {attrs[2]} ) AND {attrs[3]} ) OR ( {attrs[4]} AND {attrs[5]} )'
    
    def test_keygen_scalability(self, system_setup, heartbeat, oom_protection):
        """Key Generation scalability test (adaptive repeat count)"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        
        results = {}
        total_steps = len(self.SCALES)
        step = 0
        
        for scale in self.SCALES:
            gc.collect()
            attrs = [f'attr_{i}' for i in range(scale)]
            num_reps = get_repeats(scale)
            
            times = []
            for rep in range(num_reps):
                start = time.time()
                SK = tcabe.keygen(MSK, attrs)
                times.append((time.time() - start) * 1000)
                assert SK is not None
            
            sorted_times = sorted(times)
            p95_idx = int(len(sorted_times) * 0.95)
            results[scale] = {
                'mean_ms': round(statistics.mean(times), 2),
                'std_ms': round(statistics.stdev(times), 2) if len(times) > 1 else 0,
                'min_ms': round(min(times), 2),
                'max_ms': round(max(times), 2),
                'p95_ms': round(sorted_times[p95_idx], 2),
                'num_repeats': len(times),
            }
            
            step += 1
            heartbeat(step, total_steps,
                      f'KeyGen: {scale} attrs → {results[scale]["mean_ms"]:.1f}ms (±{results[scale]["std_ms"]:.1f}ms)')
            oom_protection.check_and_protect()
        
        self._save_scalability_data(results, 'keygen_scalability.json')
        
        print("\n" + "=" * 80)
        print("Key Generation Scalability (used for SCI paper Figure 1)")
        print("=" * 80)
        for scale in self.SCALES:
            r = results[scale]
            print(f"{scale:6}  Attributes → mean={r['mean_ms']/1000:8.3f}s (±{r['std_ms']/1000:.3f}s) [{r['num_repeats']} reps]")
    
    def test_encryption_scalability(self, system_setup, heartbeat, oom_protection):
        """Encryption scalability test (with policy depth variation)"""
        tcabe = system_setup['tcabe']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        # Policy depth test points
        POLICY_DEPTHS = [2, 5, 8, 10, 15, 20]
        
        results = {}
        total_steps = len(self.SCALES) + len(POLICY_DEPTHS)
        step = 0
        
        # Part 1: Different attribute counts (fixed policy depth)
        for scale in self.SCALES:
            gc.collect()
            
            # Simple policy encryption
            simple_policy = self._generate_simple_policy(scale)
            policy_tree = parser.parse(simple_policy)
            M = group.random(GT)
            
            start = time.time()
            CT = tcabe.encrypt(M, policy_tree)
            simple_time = time.time() - start
            assert CT is not None
            
            # Complex policy encryption
            complex_policy = self._generate_complex_policy(scale)
            policy_tree = parser.parse(complex_policy)
            
            start = time.time()
            CT = tcabe.encrypt(M, policy_tree)
            complex_time = time.time() - start
            assert CT is not None
            
            results[scale] = {
                'simple': simple_time,
                'complex': complex_time
            }
            
            step += 1
            heartbeat(step, total_steps, f'Encrypt (attrs={scale}): Simple={simple_time:.3f}s, Complex={complex_time:.3f}s')
            oom_protection.check_and_protect()
        
        # Part 2: Different policy depths (fixed attribute count, multiple repeat measurements)
        depth_results = {}
        DEPTH_REPEATS = 10  # Policy depth test repeat count
        
        for depth in POLICY_DEPTHS:
            gc.collect()
            
            # Generate nested policy
            policy = 'attr_0'
            for i in range(1, depth):
                policy = f'( {policy} AND attr_{i} )'
            
            policy_tree = parser.parse(policy)
            
            # Multiple repeat measurements
            times = []
            for rep in range(DEPTH_REPEATS):
                M = group.random(GT)
                start = time.time()
                CT = tcabe.encrypt(M, policy_tree)
                times.append(time.time() - start)
                assert CT is not None
            
            depth_results[depth] = {
                'mean_ms': round(statistics.mean(times) * 1000, 2),
                'std_ms': round(statistics.stdev(times) * 1000, 2) if len(times) > 1 else 0,
                'min_ms': round(min(times) * 1000, 2),
                'max_ms': round(max(times) * 1000, 2),
                'num_repeats': DEPTH_REPEATS
            }
            
            step += 1
            heartbeat(step, total_steps, f'Encrypt (depth={depth}): {depth_results[depth]["mean_ms"]:.2f}ms (±{depth_results[depth]["std_ms"]:.2f}ms)')
            oom_protection.check_and_protect()
        
        # Merge results
        results['by_depth'] = depth_results
        
        self._save_scalability_data(results, 'encryption_scalability.json')
        
        print("\n" + "=" * 80)
        print("Encryption Time Scalability (used for SCI paper Figure 2)")
        print("=" * 80)
        print("By Attribute Count:")
        for scale in self.SCALES:
            print(f"{scale:6} attrs - Simple: {results[scale]['simple']:8.3f}s, " +
                  f"Complex: {results[scale]['complex']:8.3f}s")
        print("\nBy Policy Depth (10 repeat measurements):")
        for depth in POLICY_DEPTHS:
            r = depth_results[depth]
            print(f"Depth {depth:2}: mean={r['mean_ms']:8.2f}ms (±{r['std_ms']:.2f}ms)")
    
    def test_decryption_scalability(self, system_setup, heartbeat, oom_protection):
        """Decryption scalability test (with policy depth variation)"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        # Policy depth test points
        POLICY_DEPTHS = [2, 5, 8, 10, 15, 20]
        
        results = {}
        total_steps = len(self.SCALES) + len(POLICY_DEPTHS)
        step = 0
        
        # Part 1: Different attribute counts (fixed policy depth)
        for scale in self.SCALES:
            gc.collect()
            
            attrs = [f'attr_{i}' for i in range(scale)]
            SK = tcabe.keygen(MSK, attrs)
            
            # Simple policy decryption
            simple_policy = self._generate_simple_policy(scale)
            policy_tree = parser.parse(simple_policy)
            M = group.random(GT)
            CT = tcabe.encrypt(M, policy_tree)
            
            start = time.time()
            decrypted = tcabe.decrypt(SK, CT)
            simple_time = time.time() - start
            assert decrypted == M
            
            # Complex policy decryption
            complex_policy = self._generate_complex_policy(scale)
            policy_tree = parser.parse(complex_policy)
            CT = tcabe.encrypt(M, policy_tree)
            
            start = time.time()
            decrypted = tcabe.decrypt(SK, CT)
            complex_time = time.time() - start
            assert decrypted == M
            
            results[scale] = {
                'simple': simple_time,
                'complex': complex_time
            }
            
            step += 1
            heartbeat(step, total_steps, f'Decrypt (attrs={scale}): Simple={simple_time:.3f}s, Complex={complex_time:.3f}s')
            oom_protection.check_and_protect()
        
        # Part 2: Different policy depths (fixed attribute count, multiple repeat measurements)
        depth_results = {}
        DEPTH_REPEATS = 10  # Policy depth test repeat count
        
        for depth in POLICY_DEPTHS:
            gc.collect()
            
            # Generate nested policy and corresponding attributes
            policy = 'attr_0'
            attrs = ['attr_0']
            for i in range(1, depth):
                policy = f'( {policy} AND attr_{i} )'
                attrs.append(f'attr_{i}')
            
            policy_tree = parser.parse(policy)
            SK = tcabe.keygen(MSK, attrs)
            
            # Multiple repeat measurements
            times = []
            for rep in range(DEPTH_REPEATS):
                M = group.random(GT)
                CT = tcabe.encrypt(M, policy_tree)
                
                start = time.time()
                decrypted = tcabe.decrypt(SK, CT)
                times.append(time.time() - start)
                assert decrypted == M
            
            depth_results[depth] = {
                'mean_ms': round(statistics.mean(times) * 1000, 2),
                'std_ms': round(statistics.stdev(times) * 1000, 2) if len(times) > 1 else 0,
                'min_ms': round(min(times) * 1000, 2),
                'max_ms': round(max(times) * 1000, 2),
                'num_repeats': DEPTH_REPEATS
            }
            
            step += 1
            heartbeat(step, total_steps, f'Decrypt (depth={depth}): {depth_results[depth]["mean_ms"]:.2f}ms (±{depth_results[depth]["std_ms"]:.2f}ms)')
            oom_protection.check_and_protect()
        
        # Merge results
        results['by_depth'] = depth_results
        
        self._save_scalability_data(results, 'decryption_scalability.json')
        
        print("\n" + "=" * 80)
        print("Decryption Time Scalability")
        print("=" * 80)
        print("By Attribute Count:")
        for scale in self.SCALES:
            print(f"{scale:6} attrs - Simple: {results[scale]['simple']:8.3f}s, " +
                  f"Complex: {results[scale]['complex']:8.3f}s")
        print("\nBy Policy Depth (10 repeat measurements):")
        for depth in POLICY_DEPTHS:
            r = depth_results[depth]
            print(f"Depth {depth:2}: mean={r['mean_ms']:8.2f}ms (±{r['std_ms']:.2f}ms)")
    
    def test_batch_operation_scalability(self, system_setup, heartbeat, oom_protection):
        """Batch operation scalability test"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        batch_sizes = [10, 50, 100, 500, 1000]
        results = {}
        
        policy_str = 'role:engineer AND dept:maintenance'
        policy_tree = parser.parse(policy_str)
        attrs = ['role:engineer', 'dept:maintenance']
        SK = tcabe.keygen(MSK, attrs)
        
        total_steps = len(batch_sizes) * 2
        step = 0
        
        for batch_size in batch_sizes:
            gc.collect()
            messages = [group.random(GT) for _ in range(batch_size)]
            
            # Batch encryption
            start = time.time()
            ciphertexts = []
            for M in messages:
                CT = tcabe.encrypt(M, policy_tree)
                ciphertexts.append(CT)
            encrypt_time = time.time() - start
            assert len(ciphertexts) == batch_size
            
            step += 1
            heartbeat(step, total_steps, f'Batch Encrypt: {batch_size} → {encrypt_time:.3f}s')
            
            # Batch decryption
            start = time.time()
            decrypted_list = []
            for CT in ciphertexts:
                decrypted = tcabe.decrypt(SK, CT)
                decrypted_list.append(decrypted)
            decrypt_time = time.time() - start
            assert len(decrypted_list) == batch_size
            assert all(d == m for d, m in zip(decrypted_list, messages))
            
            results[batch_size] = {
                'encrypt': encrypt_time,
                'decrypt': decrypt_time
            }
            
            step += 1
            heartbeat(step, total_steps, f'Batch Decrypt: {batch_size} → {decrypt_time:.3f}s')
            oom_protection.check_and_protect()
        
        self._save_scalability_data(results, 'batch_scalability.json')
        
        print("\n" + "=" * 80)
        print("Batch Operation Scalability")
        print("=" * 80)
        for batch_size in batch_sizes:
            print(f"{batch_size:6} items - Encrypt: {results[batch_size]['encrypt']:8.3f}s, " +
                  f"Decrypt: {results[batch_size]['decrypt']:8.3f}s")
    
    def test_digital_twin_scalability(self, system_setup, heartbeat, oom_protection):
        """Digital Twin management scalability test"""
        dt_manager = system_setup['dt_manager']
        
        twin_scales = [10, 50, 100, 500, 1000]
        results = {}
        total_steps = len(twin_scales) * 2
        step = 0
        
        for num_twins in twin_scales:
            gc.collect()
            
            # Create twins
            start = time.time()
            created = []
            for i in range(num_twins):
                twin_id = f'twin_{step}_{i}'
                result = dt_manager.create_digital_twin(twin_id, {
                    'type': 'sensor',
                    'index': i,
                    'location': f'factory_{i % 10}'
                })
                assert result['success']
                created.append(twin_id)
            create_time = time.time() - start
            
            step += 1
            heartbeat(step, total_steps, f'Create Twins: {num_twins} → {create_time:.3f}s')
            
            # Send commands
            start = time.time()
            for twin_id in created:
                result = dt_manager.send_command(twin_id, {
                    'action': 'read_sensor',
                    'params': {'type': 'temperature'}
                })
                assert result['success']
            command_time = time.time() - start
            
            results[num_twins] = {
                'create': create_time,
                'command': command_time
            }
            
            step += 1
            heartbeat(step, total_steps, f'Command Twins: {num_twins} → {command_time:.3f}s')
            oom_protection.check_and_protect()
        
        self._save_scalability_data(results, 'digital_twin_scalability.json')
    
    def test_policy_depth_scalability(self, system_setup, heartbeat, oom_protection):
        """Policy tree depth scalability test"""
        tcabe = system_setup['tcabe']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        MSK = system_setup['MSK']
        
        depths = [1, 2, 3, 5, 8]
        results = {}
        total_steps = len(depths)
        step = 0
        
        for depth in depths:
            gc.collect()
            
            # Generate nested policy
            policy = 'attr_0'
            for i in range(1, depth):
                policy = f'( {policy} AND attr_{i} )'
            
            policy_tree = parser.parse(policy)
            attrs = [f'attr_{i}' for i in range(depth)]
            SK = tcabe.keygen(MSK, attrs)
            M = group.random(GT)
            
            start = time.time()
            CT = tcabe.encrypt(M, policy_tree)
            encrypt_time = time.time() - start
            assert CT is not None
            
            start = time.time()
            decrypted = tcabe.decrypt(SK, CT)
            decrypt_time = time.time() - start
            assert decrypted == M
            
            results[depth] = {
                'encrypt': encrypt_time,
                'decrypt': decrypt_time
            }
            
            step += 1
            heartbeat(step, total_steps, f'Policy depth {depth}: Encrypt {encrypt_time:.3f}s, Decrypt {decrypt_time:.3f}s')
            oom_protection.check_and_protect()
        
        self._save_scalability_data(results, 'policy_depth_scalability.json')
    
    def test_generate_scalability_table(self, system_setup, heartbeat, oom_protection):
        """Generate scalability test table (used for SCI paper Table 2) - multiple rounds for mean value to eliminate noise"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']

        # Rounds per scale: more rounds for small scale to eliminate noise, fewer rounds for large scale to control total time
        rounds_per_scale = {10: 20, 50: 20, 100: 20, 500: 10, 1000: 10, 5000: 5, 10000: 5}

        results = {}
        total_steps = sum(rounds_per_scale[s] for s in self.SCALES)
        step = 0

        for scale in self.SCALES:
            gc.collect()
            N_ROUNDS = rounds_per_scale[scale]
            keygen_times = []
            encrypt_times = []
            decrypt_times = []

            attrs = [f'attr_{i}' for i in range(scale)]
            policy_str = self._generate_simple_policy(scale)
            policy_tree = parser.parse(policy_str)

            for r in range(N_ROUNDS):
                M = group.random(GT)

                start = time.time()
                SK = tcabe.keygen(MSK, attrs)
                keygen_times.append((time.time() - start) * 1000)
                assert SK is not None

                start = time.time()
                CT = tcabe.encrypt(M, policy_tree)
                encrypt_times.append((time.time() - start) * 1000)
                assert CT is not None

                start = time.time()
                decrypted = tcabe.decrypt(SK, CT)
                decrypt_times.append((time.time() - start) * 1000)
                assert decrypted == M

                step += 1
                heartbeat(step, total_steps, f'Scale {scale} round {r+1}/{N_ROUNDS}')
                oom_protection.check_and_protect()

            kg_mean = round(sum(keygen_times) / len(keygen_times), 2)
            enc_mean = round(sum(encrypt_times) / len(encrypt_times), 2)
            dec_mean = round(sum(decrypt_times) / len(decrypt_times), 2)
            throughput = round(1000.0 / dec_mean, 2) if dec_mean > 0 else 0

            results[scale] = {
                'keygen_ms': kg_mean,
                'encrypt_ms': enc_mean,
                'decrypt_ms': dec_mean,
                'decrypt_std': round((sum((x - dec_mean)**2 for x in decrypt_times) / len(decrypt_times))**0.5, 2),
                'throughput_ops': throughput
            }

        self._save_scalability_data(results, 'scalability_summary_table.json')

        print("\n" + "=" * 100)
        print("Table 2: Large Scale Scalability Test Results (used for SCI paper)")
        print("=" * 100)
        print(f"{'Scale':12} {'KeyGen(ms)':12} {'Encrypt(ms)':12} {'Decrypt(ms)':12} {'Throughput(op/s)':18}")
        print("-" * 100)
        for scale in self.SCALES:
            r = results[scale]
            print(f"{scale:12} {r['keygen_ms']:11.1f} {r['encrypt_ms']:11.1f} {r['decrypt_ms']:11.1f} {r['throughput_ops']:16.1f}")
        print("=" * 100)
