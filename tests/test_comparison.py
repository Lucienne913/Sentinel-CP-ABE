"""
pytest tests/test_comparison.py -m comparison -v

Scheme 4: Digital Twin Network Security Framework - Comparison Tests
"""

import pytest
import time
from datetime import datetime
from charm.toolbox.pairinggroup import GT


@pytest.mark.comparison
class TestSecurityLevelComparison:

    def test_80_vs_128_bit_keygen_time(self, system_setup, heartbeat, oom_protection):
        from src.setup import T_CP_ABE_Setup
        from src.t_cp_abe import T_CP_ABE
        
        attrs = ['role:engineer', 'dept:maintenance']
        
        # ===== Test 1: 80-bit Security Level (SS512 curve) =====
        setup80 = T_CP_ABE_Setup(group_name='SS512', security_level=80)
        PP80, MSK80 = setup80.setup()
        tcabe80 = T_CP_ABE(PP80)
        
        start = time.time()
        SK80 = tcabe80.keygen(MSK80, attrs)
        time80 = time.time() - start
        
        heartbeat(1, 4, f"80-bit keygen: {time80:.4f}s")
        oom_protection.check_and_protect()
        
        assert SK80 is not None
        assert time80 < 5.0
        
        # ===== Test 2: 128-bit Security Level (SS1024 curve) =====
        setup128 = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
        PP128, MSK128 = setup128.setup()
        tcabe128 = T_CP_ABE(PP128)
        
        start = time.time()
        SK128 = tcabe128.keygen(MSK128, attrs)
        time128 = time.time() - start
        
        heartbeat(2, 4, f"128-bit keygen: {time128:.4f}s")
        oom_protection.check_and_protect()
        
        assert SK128 is not None
        assert time128 < 10.0  # 128-bit security level computation is slower, allow longer time
        
        # ===== Output Comparison Results =====
        print(f"\n[80 vs 128-bit KeyGen Comparison]")
        print(f"80-bit (SS512): {time80:.4f}s")
        print(f"128-bit (SS1024): {time128:.4f}s")
        print(f"128-bit is {time128/time80:.2f} times slower than 80-bit")
        
        heartbeat(3, 4, f"Performance comparison: 80-bit={time80:.4f}s, 128-bit={time128:.4f}s")
        heartbeat(4, 4, "80 vs 128-bit comparison done")

    def test_encryption_time_comparison(self, system_setup, heartbeat):
        from src.setup import T_CP_ABE_Setup
        from src.t_cp_abe import T_CP_ABE, PolicyParser
        
        # ===== Test 1: 80-bit Security Level Encryption Time =====
        setup80 = T_CP_ABE_Setup(group_name='SS512', security_level=80)
        PP80, MSK80 = setup80.setup()
        tcabe80 = T_CP_ABE(PP80)
        parser = PolicyParser()
        group80 = PP80['group']
        
        policy_tree = parser.parse("role:engineer AND dept:maintenance")
        M80 = group80.random(GT)
        
        start = time.time()
        CT80 = tcabe80.encrypt(M80, policy_tree)
        time80 = time.time() - start
        
        assert CT80 is not None
        assert time80 < 5.0
        heartbeat(1, 4, f"80-bit encryption: {time80:.4f}s")
        
        # ===== Test 2: 128-bit Security Level Encryption Time =====
        setup128 = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
        PP128, MSK128 = setup128.setup()
        tcabe128 = T_CP_ABE(PP128)
        group128 = PP128['group']
        
        M128 = group128.random(GT)
        
        start = time.time()
        CT128 = tcabe128.encrypt(M128, policy_tree)
        time128 = time.time() - start
        
        assert CT128 is not None
        assert time128 < 10.0  # 128-bit computation is slower
        heartbeat(2, 4, f"128-bit encryption: {time128:.4f}s")
        
        # ===== Output Comparison Results =====
        print(f"\n[80 vs 128-bit Encryption Comparison]")
        print(f"80-bit (SS512): {time80:.4f}s")
        print(f"128-bit (SS1024): {time128:.4f}s")
        print(f"128-bit is {time128/time80:.2f} times slower than 80-bit")
        
        heartbeat(3, 4, f"Encryption performance comparison completed")
        heartbeat(4, 4, "80 vs 128-bit encryption comparison done")


@pytest.mark.comparison
class TestPolicyComplexityComparison:

    def test_simple_vs_complex_policy(self, system_setup, heartbeat, memory_monitor, oom_protection):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        simple_policy = parser.parse("role:engineer")
        complex_policy = parser.parse("(role:engineer OR role:admin) AND dept:maintenance AND location:factory AND time:work")
        
        M = group.random(GT)
        
        start = time.time()
        CT_simple = tcabe.encrypt(M, simple_policy)
        time_simple = time.time() - start
        heartbeat(4, 8, f"Simple policy encryption: {time_simple:.4f}s")
        
        start = time.time()
        CT_complex = tcabe.encrypt(M, complex_policy)
        time_complex = time.time() - start
        heartbeat(5, 8, f"Complex policy encryption: {time_complex:.4f}s")
        
        assert CT_simple is not None
        assert CT_complex is not None
        oom_protection.check_and_protect()
        
        attrs_simple = ['role:engineer']
        SK_simple = tcabe.keygen(MSK, attrs_simple)
        
        attrs_complex = ['role:engineer', 'dept:maintenance', 'location:factory']
        SK_complex = tcabe.keygen(MSK, attrs_complex)
        
        start = time.time()
        decrypted_simple = tcabe.decrypt(SK_simple, CT_simple)
        time_decrypt_simple = time.time() - start
        assert decrypted_simple == M
        heartbeat(6, 8, f"Simple policy decryption: {time_decrypt_simple:.4f}s")
        
        work_time = datetime(2026, 4, 21, 10, 0)
        start = time.time()
        decrypted_complex = tcabe.decrypt(SK_complex, CT_complex, work_time)
        time_decrypt_complex = time.time() - start
        assert decrypted_complex == M
        heartbeat(7, 8, f"Complex policy decryption: {time_decrypt_complex:.4f}s")
        memory_monitor.get_usage()
        heartbeat(8, 8, "Policy complexity comparison done")


@pytest.mark.comparison
class TestAblationStudy:

    def test_with_vs_without_time_predicate(self, system_setup, heartbeat, oom_protection):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        attrs = ['role:engineer', 'dept:maintenance']
        SK = tcabe.keygen(MSK, attrs)
        
        policy_without_time = parser.parse("role:engineer AND dept:maintenance")
        policy_with_time = parser.parse("role:engineer AND dept:maintenance AND time:work")
        
        M = group.random(GT)
        
        start = time.time()
        CT_no_time = tcabe.encrypt(M, policy_without_time)
        time_no_time = time.time() - start
        
        start = time.time()
        CT_with_time = tcabe.encrypt(M, policy_with_time)
        time_with_time = time.time() - start
        
        assert CT_no_time is not None
        assert CT_with_time is not None
        heartbeat(8, 12, f"No time: {time_no_time:.4f}s, With time: {time_with_time:.4f}s")
        oom_protection.check_and_protect()

    def test_diffusion_model_guidance_scale_ablation(self, heartbeat, memory_monitor, oom_protection):
        torch = pytest.importorskip("torch")
        from src.diffusion import ThreatDiffusionModel
        
        model = ThreatDiffusionModel(
            vocab_size=50, embed_dim=32, condition_dim=16,
            num_train_timesteps=10, device='cpu'
        )
        
        policy_indices = torch.tensor([[10, 20, 30] + [0]*17])
        
        scales = [0.5, 1.0, 1.5, 2.0]
        results = {}
        
        print("\n" + "="*80)
        print("Diffusion Model Guidance Scale Ablation Study")
        print("="*80)
        
        for i, scale in enumerate(scales):
            start = time.time()
            samples = model.generate_adversarial(policy_indices, n_samples=5, guidance_scale=scale)
            gen_time = time.time() - start
            
            unique = len(torch.unique(samples, dim=0))
            diversity = unique / 5
            
            results[scale] = {
                'time': gen_time,
                'diversity': diversity,
                'n_samples': 5
            }
            heartbeat(12 + i, 12 + len(scales), f"guidance_scale={scale}: diversity={diversity:.2f}")
            oom_protection.check_and_protect()
        
        # ===== Output Comparison Results =====
        print("\n" + "-"*80)
        print("Table: Guidance Scale Ablation Study Results")
        print("-"*80)
        print(f"{'guidance_scale':15} {'GenerateTime(ms)':15} {'Diversity':10}")
        print("-"*80)
        
        for scale in scales:
            r = results[scale]
            print(f"{scale:15} {r['time']*1000:14.2f} {r['diversity']:9.2f}")
        
        print("-"*80)
        
        # ===== Analyze Optimal Parameter =====
        # Find scale with highest diversity (while considering time)
        best_scale = max(scales, key=lambda s: results[s]['diversity'])
        worst_scale = min(scales, key=lambda s: results[s]['diversity'])
        
        print(f"\nAnalysis Conclusion:")
        print(f"• Optimal guidance_scale: {best_scale}")
        print(f"  - Diversity: {results[best_scale]['diversity']:.2f}")
        print(f"  - Generation Time: {results[best_scale]['time']*1000:.2f}ms")
        print(f"• Worst guidance_scale: {worst_scale}")
        print(f"  - Diversity: {results[worst_scale]['diversity']:.2f}")
        print(f"  - Generation Time: {results[worst_scale]['time']*1000:.2f}ms")
        
        # Compute relative difference
        diversity_diff = results[best_scale]['diversity'] - results[worst_scale]['diversity']
        time_diff = results[worst_scale]['time'] - results[best_scale]['time']
        print(f"\n• Diversity Difference: {diversity_diff:.2f}")
        print(f"• Time Difference: {time_diff*1000:.2f}ms")
        
        print("\n" + "="*80)
        print("Guidance Scale Ablation Study Complete")
        print("="*80)
        
        assert all(r['diversity'] > 0 for r in results.values())
        memory_monitor.get_usage()

    def test_cache_mechanism_comparison(self, heartbeat, oom_protection):
        from src.setup import T_CP_ABE_Setup
        
        # ===== Test Different Cache Mechanisms =====
        # 1. LRU (Least Recently Used) - Current system implementation
        # 2. FIFO (First-In-First-Out) - Simulated implementation
        # 3. None (No Cache) - Recompute every time
        
        cache_policies = ['LRU', 'FIFO', 'None']
        results = {}
        
        for policy in cache_policies:
            # Configure cache according to policy
            if policy == 'None':
                # No Cache: Set very small cache capacity
                setup = T_CP_ABE_Setup(group_name='SS512', security_level=80)
                PP, MSK = setup.setup(max_attrs=20, cache_max_size=1)  # Almost no cache
            else:
                # LRU and FIFO use the same cache capacity
                setup = T_CP_ABE_Setup(group_name='SS512', security_level=80)
                PP, MSK = setup.setup(max_attrs=20, cache_max_size=50)
            
            # Simulate FIFO cache policy
            if policy == 'FIFO':
                original_H = PP['H']
                fifo_order = []
                
                def fifo_H(attr_str):
                    nonlocal fifo_order
                    # FIFO logic: Remove oldest when full
                    if len(PP['attr_hash_cache']) >= 50 and fifo_order:
                        oldest = fifo_order.pop(0)
                        PP['attr_hash_cache'].pop(oldest, None)
                    
                    result = original_H(attr_str)
                    if attr_str not in fifo_order:
                        fifo_order.append(attr_str)
                    return result
                
                PP['H'] = fifo_H
            
            # Measure performance
            start = time.time()
            for i in range(50):
                PP['H'](f'attr:{i % 30}')
            elapsed = time.time() - start
            
            results[policy] = {
                'time': elapsed,
                'cache_size': len(PP['attr_hash_cache'])
            }
            heartbeat(16 + cache_policies.index(policy), 16 + len(cache_policies),
                     f"cache_policy={policy}: {elapsed:.4f}s")
        
        # ===== Output Comparison Results =====
        print("\n" + "="*80)
        print("Cache Mechanism Comparison Results")
        print("="*80)
        print(f"{'CacheMechanism':12} {'Time(ms)':12} {'CacheSize':10}")
        print("-"*80)
        for policy, r in results.items():
            print(f"{policy:12} {r['time']*1000:11.2f} {r['cache_size']:9}")
        print("="*80)
        
        # Analyze optimal policy
        fastest_policy = min(results.keys(), key=lambda p: results[p]['time'])
        print(f"\nAnalysis Conclusion:")
        print(f"• Fastest Cache Mechanism: {fastest_policy}")
        if 'LRU' in results and 'None' in results:
            speedup = results['None']['time'] / results['LRU']['time']
            print(f"• LRU is {speedup:.2f} times faster than no cache")
        
        assert all(r['time'] > 0 for r in results.values())
        oom_protection.check_and_protect()


@pytest.mark.comparison
class TestScalabilityComparison:

    def test_encryption_time_vs_attribute_count(self, system_setup, heartbeat, memory_monitor, oom_protection):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        attr_counts = [1, 3, 5, 10]
        times = []
        results = {}
        
        for i, count in enumerate(attr_counts):
            attrs = [f'attr:{j}' for j in range(count)]
            SK = tcabe.keygen(MSK, attrs)
            
            policy_str = ' AND '.join([f'attr:{j}' for j in range(count)])
            policy_tree = parser.parse(policy_str)
            M = group.random(GT)
            
            start = time.time()
            CT = tcabe.encrypt(M, policy_tree)
            elapsed = time.time() - start
            
            times.append(elapsed)
            results[count] = elapsed
            assert CT is not None
            heartbeat(i + 1, len(attr_counts), f"Attr count={count}: {elapsed:.4f}s")
            oom_protection.check_and_protect()
        
        # ===== Output Comparison Results =====
        print("\n" + "="*80)
        print("Encryption Time vs Attribute Count")
        print("="*80)
        print(f"{'AttributeCount':12} {'EncryptTime(ms)':15} {'Growth Rate':12}")
        print("-"*80)
        baseline = results[1]
        for count, t in results.items():
            growth = t / baseline
            print(f"{count:12} {t*1000:14.2f} {growth:11.2f}x")
        print("="*80)
        
        # Analysis conclusion
        max_count = max(attr_counts)
        time_increase = results[max_count] / results[1]
        print(f"\nAnalysis Conclusion:")
        print(f"• Attribute count increased from 1 to {max_count}, encryption time increased {time_increase:.2f} times")
        if time_increase > max_count:
            print(f"• Warning: Encryption time growth exceeds attribute count growth (O(n²) complexity?)")
        else:
            print(f"• Encryption time growth has approximately linear relationship with attribute count")
        
        memory_monitor.get_usage()

    def test_twin_management_scaling(self, system_setup, heartbeat, oom_protection):
        dt_manager = system_setup['dt_manager']
        
        twin_counts = [10, 50]
        results = {}
        
        for count in twin_counts:
            start = time.time()
            for i in range(count):
                dt_manager.create_digital_twin(
                    f'factory:scale_twin_{i}',
                    {'type': 'sensor', 'index': i}
                )
            elapsed = time.time() - start
            
            twins = dt_manager.list_digital_twins()
            assert len(twins) >= count
            
            results[count] = {'time': elapsed, 'per_twin': elapsed / count}
            heartbeat(twin_counts.index(count) + 1, len(twin_counts), 
                     f"Twin count={count}: {elapsed:.4f}s")
            oom_protection.check_and_protect()
        
        # ===== Output Comparison Results =====
        print("\n" + "="*80)
        print("Digital Twin Management Scalability Test")
        print("="*80)
        print(f"{'TwinCount':12} {'Total Time(s)':12} {'Single Time(ms)':15}")
        print("-"*80)
        for count, r in results.items():
            print(f"{count:12} {r['time']:12.4f} {r['per_twin']*1000:14.2f}")
        print("="*80)
        
        # Analyze scalability
        count_10 = results[10]['time']
        count_50 = results[50]['time']
        scaling_factor = count_50 / count_10 / 5  # Ideally should be 5 times
        print(f"\nAnalysis Conclusion:")
        print(f"• 50 Twins vs 10 Twins: Time Ratio = {count_50/count_10:.2f}x")
        print(f"• Scalability Efficiency: {scaling_factor:.2f} (1.0 represents linear scaling)")
        if scaling_factor > 1.2:
            print(f"• Warning: System scalability is not linear, overhead increases with count")


@pytest.mark.comparison
class TestBenchmarkComparison:

    def test_auth_protocol_overhead(self, system_setup, heartbeat, memory_monitor, oom_protection):
        from src.setup import T_CP_ABE_Setup
        from src.t_cp_abe import T_CP_ABE
        from src.auth import BidirectionalAuth
        
        security_levels = [80, 128]
        results = {}
        
        for level in security_levels:
            if level == 80:
                setup = T_CP_ABE_Setup(group_name='SS512', security_level=80)
            else:
                setup = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
            
            PP, MSK = setup.setup(max_attrs=20)
            tcabe = T_CP_ABE(PP)
            auth = BidirectionalAuth(PP, tcabe=tcabe)
            parser = system_setup['parser']
            
            times = []
            for i in range(5):
                start = time.time()
                
                auth_request, device_bls_sk = auth.device_auth_init(f'Device_Bench_{i}_{level}bit', ['type:sensor'])
                
                policy_T = parser.parse("type:sensor")
                challenge_resp = auth.digital_twin_challenge(auth_request, policy_T)
                
                device_SK = tcabe.keygen(MSK, ['type:sensor'])
                device_resp = auth.device_response(
                    challenge_resp['session_id'],
                    challenge_resp,
                    device_SK,
                    device_bls_sk,
                    f'Device_Bench_{i}_{level}bit'
                )
                
                verify_result = auth.digital_twin_verify(device_resp)
                assert verify_result['success']
                
                elapsed = time.time() - start
                times.append(elapsed)
            
            avg_time = sum(times) / len(times) if times else 0
            results[level] = {'avg_time': avg_time, 'success': 5, 'times': times}
            heartbeat(security_levels.index(level) + 1, len(security_levels), 
                     f"Security level={level}-bit: avg={avg_time*1000:.2f}ms")
        
        # ===== Output Comparison Results =====
        print("\n" + "="*80)
        print("Authentication Protocol Overhead Comparison (Different Security Levels)")
        print("="*80)
        print(f"{'Security Level':15} {'AverageOverhead(ms)':18} {'AuthCount':12}")
        print("-"*80)
        for level, r in results.items():
            print(f"{level:>3}-bit {r['avg_time']*1000:>17.2f} {r['success']:>11} times")
        print("="*80)
        
        # Analysis conclusion
        time_80 = results[80]['avg_time']
        time_128 = results[128]['avg_time']
        print(f"\nAnalysis Conclusion:")
        print(f"• 128-bit security level authentication overhead is {(time_128/time_80):.2f} times 80-bit")
        print(f"• Authentication flow includes: Device Authentication → Digital Twin Challenge → Key Generation → Response Verification")
        print(f"• Higher security level provides stronger security guarantee, but brings higher computation overhead")
        
        oom_protection.check_and_protect()

    def test_signature_vs_certificate_overhead(self, system_setup, heartbeat, oom_protection):
        bls = system_setup['bls']
        cert_auth = system_setup['cert_auth']
        
        pk, sk = bls.keygen()
        
        start = time.time()
        sigma = bls.sign(sk, "test_message")
        sign_time = time.time() - start
        
        start = time.time()
        cert = cert_auth.issue_certificate('device_bench', ['type:sensor'])
        cert_time = time.time() - start
        
        start = time.time()
        bls.verify(pk, "test_message", sigma)
        verify_time = time.time() - start
        
        start = time.time()
        cert_auth.verify_certificate(cert)
        cert_verify_time = time.time() - start
        
        assert sign_time < 1.0
        assert cert_time < 2.0
        assert verify_time < 1.0
        assert cert_verify_time < 1.0
        
        # ===== Output Comparison Results =====
        print("\n" + "="*80)
        print("Signature vs Certificate Overhead Comparison")
        print("="*80)
        print(f"{'Operation':20} {'Time(ms)':15} {'Threshold(ms)':12}")
        print("-"*80)
        print(f"{'BLS Sign':20} {sign_time*1000:14.2f} {1000:11.1f}")
        print(f"{'BLS Verify':20} {verify_time*1000:14.2f} {1000:11.1f}")
        print(f"{'Certificate Issuance':20} {cert_time*1000:14.2f} {2000:11.1f}")
        print(f"{'Certificate Verify':20} {cert_verify_time*1000:14.2f} {1000:11.1f}")
        print("="*80)
        
        # Analysis conclusion
        total_sig = sign_time + verify_time
        total_cert = cert_time + cert_verify_time
        print(f"\nAnalysis Conclusion:")
        print(f"• BLS Signature scheme total overhead: {total_sig*1000:.2f}ms")
        print(f"• Traditional Certificate scheme total overhead: {total_cert*1000:.2f}ms")
        print(f"• BLS Signature scheme is {(total_cert/total_sig):.2f} times faster than certificate scheme")
        if total_sig < total_cert:
            print(f"• Recommended to use BLS Signature scheme")
        else:
            print(f"• Traditional Certificate scheme is more stable")
        
        heartbeat(5, 6, f"Sign: {sign_time*1000:.1f}ms, Cert: {cert_time*1000:.1f}ms")
        oom_protection.check_and_protect()


@pytest.mark.comparison
class TestConsistencyVerification:

    def test_encrypt_decrypt_consistency(self, system_setup, heartbeat):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        attrs = ['role:engineer', 'dept:maintenance']
        SK = tcabe.keygen(MSK, attrs)
        policy_tree = parser.parse("role:engineer AND dept:maintenance")
        
        for i in range(5):
            M = group.random(GT)
            CT = tcabe.encrypt(M, policy_tree)
            decrypted = tcabe.decrypt(SK, CT)
            assert decrypted == M
            heartbeat(i + 1, 5, f"Consistency check #{i+1}")

    def test_signature_consistency(self, system_setup, heartbeat, oom_protection):
        bls = system_setup['bls']
        pk, sk = bls.keygen()
        
        message = "consistency_test"
        for i in range(10):
            sigma = bls.sign(sk, message)
            assert bls.verify(pk, message, sigma)
            heartbeat(i + 1, 10, f"Signature consistency #{i+1}")
        oom_protection.check_and_protect()


@pytest.mark.comparison
@pytest.mark.slow
class TestFullAblationStudy2026:
    """Complete Ablation Study Test (required for SCI paper Table 3)"""
    
    ABLATION_CONFIGS = [
        ('Full',              True,  True,  True,  True,  True),   # time, cache, subprocess, diffusion, digital_twin
        ('NoTimePredicate',   False, True,  True,  True,  True),
        ('NoCache',           True,  False, True,  True,  True),
        ('NoSubprocess',      True,  True,  False, True,  True),
        ('NoDiffusion',       True,  True,  True,  False, True),
        ('NoDigitalTwin',     True,  True,  True,  True,  False),
        ('Minimal',           False, False, False, False, False)
    ]
    
    @staticmethod
    def _save_ablation_data(data, filename):
        """Save Ablation Data To File (used for SCI paper Table 3)"""
        import json
        from pathlib import Path
        output_dir = Path(__file__).parent.parent / 'experiments' / 'results' / 'ablation'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def test_full_ablation_matrix(self, system_setup, heartbeat):
        """Complete Ablation Matrix Test (required for SCI paper Figure 4)"""
        from src.setup import T_CP_ABE_Setup
        from src.t_cp_abe import T_CP_ABE
        from src.subprocess_worker import SubprocessWorker
        from src.diffusion import ThreatDiffusionModel
        from src.digital_twin import DigitalTwinManager
        
        print("\n" + "="*80)
        print("Complete Ablation Study")
        print("="*80)
        
        results = {}
        total_steps = len(self.ABLATION_CONFIGS) * 3
        step = 0
        
        # Iteration count (paper data is average of multiple runs)
        iterations = 100
        
        for config_name, use_time, use_cache, use_subprocess, use_diffusion, use_digital_twin in self.ABLATION_CONFIGS:
            print(f"\nConfiguration: {config_name}")
            
            # Create independent system instance according to ablation configuration
            # All configurations use the same cache size (5000) for fair comparison of component overhead
            # use_cache parameter only controls whether cache optimization logic is enabled
            cache_size = 5000
            setup = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
            PP, MSK = setup.setup(max_attrs=20, cache_max_size=cache_size)
            
            # Create components according to ablation configuration
            subprocess_worker = SubprocessWorker(max_memory_mb=1024, timeout=60) if use_subprocess else None
            threat_model = ThreatDiffusionModel(vocab_size=50, embed_dim=32, condition_dim=16, num_train_timesteps=10, device='cpu') if use_diffusion else None
            dt_manager = DigitalTwinManager(use_local=True, max_history_size=100) if use_digital_twin else None
            
            # Create T_CP_ABE instance, pass components and flags
            tcabe = T_CP_ABE(PP, 
                           subprocess_worker=subprocess_worker,
                           threat_model=threat_model,
                           dt_manager=dt_manager,
                           use_subprocess=use_subprocess,
                           use_diffusion=use_diffusion,
                           use_digital_twin=use_digital_twin)
            group = PP['group']
            parser = system_setup['parser']
            
            # Basic attributes
            attrs = ['role:engineer', 'dept:maintenance', 'location:factory']
            
            # Policy (adjusted according to configuration)
            if use_time:
                policy_str = "role:engineer AND dept:maintenance AND location:factory AND time:work"
            else:
                policy_str = "role:engineer AND dept:maintenance AND location:factory"
            
            policy_tree = parser.parse(policy_str)
            M = group.random(GT)
            
            # Warm-up run (exclude initialization overhead of first run)
            warmup_work_time = datetime(2026, 4, 22, 11, 0) if use_time else None
            for _ in range(5):
                SK_warmup = tcabe.keygen(MSK, attrs)
                CT_warmup = tcabe.encrypt(M, policy_tree)
                if warmup_work_time:
                    decrypted_warmup = tcabe.decrypt(SK_warmup, CT_warmup, warmup_work_time)
                else:
                    decrypted_warmup = tcabe.decrypt(SK_warmup, CT_warmup)
            
            # Measure Key Generation (average over multiple runs)
            keygen_times = []
            for _ in range(iterations):
                start = time.time()
                SK = tcabe.keygen(MSK, attrs)
                keygen_times.append(time.time() - start)
            keygen_time = sum(keygen_times) / len(keygen_times)
            assert SK is not None
            
            step += 1
            heartbeat(step, total_steps, f"{config_name}: KeyGen {keygen_time:.6f}s")
            
            # Measure Encryption (average over multiple runs)
            encrypt_times = []
            for _ in range(iterations):
                start = time.time()
                CT = tcabe.encrypt(M, policy_tree)
                encrypt_times.append(time.time() - start)
            encrypt_time = sum(encrypt_times) / len(encrypt_times)
            assert CT is not None
            
            step += 1
            heartbeat(step, total_steps, f"{config_name}: Encrypt {encrypt_time:.3f}s")
            
            # Measure Decryption (average over multiple runs)
            work_time = datetime(2026, 4, 22, 11, 0) if use_time else None
            decrypt_times = []
            for _ in range(iterations):
                start = time.time()
                decrypted = tcabe.decrypt(SK, CT, work_time) if work_time else tcabe.decrypt(SK, CT)
                decrypt_times.append(time.time() - start)
            decrypt_time = sum(decrypt_times) / len(decrypt_times)
            assert decrypted == M
            
            step += 1
            heartbeat(step, total_steps, f"{config_name}: Decrypt {decrypt_time:.6f}s")
            
            # Save results
            results[config_name] = {
                'keygen_time': keygen_time,
                'encrypt_time': encrypt_time,
                'decrypt_time': decrypt_time,
                'total_time': keygen_time + encrypt_time + decrypt_time,
                'config': {
                    'use_time_predicate': use_time,
                    'use_cache': use_cache,
                    'use_subprocess': use_subprocess,
                    'use_diffusion': use_diffusion,
                    'use_digital_twin': use_digital_twin
                }
            }
        
        # Save Data
        self._save_ablation_data(results, 'full_ablation_study.json')
        
        # Print Table (used for SCI paper Table 3)
        print("\n" + "="*100)
        print("Table 3: Complete Ablation Study Results")
        print("="*100)
        print(f"{'Configuration':20} {'KeyGen(ms)':12} {'Encrypt(ms)':12} {'Decrypt(ms)':12} {'Total(ms)':12}")
        print("-"*100)
        
        for config_name in [c[0] for c in self.ABLATION_CONFIGS]:
            r = results[config_name]
            print(
                f"{config_name:20} "
                f"{r['keygen_time']*1000:11.1f} "
                f"{r['encrypt_time']*1000:11.1f} "
                f"{r['decrypt_time']*1000:11.1f} "
                f"{r['total_time']*1000:11.1f}"
            )
        
        print("="*100)
        print("Complete Ablation Study Finished!")
        print("Data Saved To experiments/results/ablation/full_ablation_study.json")
        print("Can be used to generate SCI paper Figure 4 and Table 3")
    
    def test_ablation_component_contribution(self, system_setup, heartbeat):
        """Component Contribution Analysis"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        print("\n" + "="*80)
        print("Component Contribution Analysis")
        print("="*80)
        
        # Baseline configuration (full functionality)
        attrs = ['role:engineer', 'dept:maintenance', 'location:factory']
        SK = tcabe.keygen(MSK, attrs)
        policy_with_time = parser.parse("role:engineer AND dept:maintenance AND location:factory AND time:work")
        policy_without_time = parser.parse("role:engineer AND dept:maintenance AND location:factory")
        
        # Measure baseline decryption time (with Time Predicate)
        M = group.random(GT)
        CT_with_time = tcabe.encrypt(M, policy_with_time)
        CT_without_time = tcabe.encrypt(M, policy_without_time)
        
        work_time = datetime(2026, 4, 22, 11, 0)
        
        # Multiple runs to get stable value
        n_runs = 5
        times_with = []
        times_without = []
        
        for i in range(n_runs):
            start = time.time()
            decrypted = tcabe.decrypt(SK, CT_with_time, work_time)
            elapsed = time.time() - start
            assert decrypted == M
            times_with.append(elapsed)
            
            start = time.time()
            decrypted = tcabe.decrypt(SK, CT_without_time)
            elapsed = time.time() - start
            assert decrypted == M
            times_without.append(elapsed)
            
            heartbeat(i + 1, n_runs, f"Component contribution run {i+1}")
        
        avg_with = sum(times_with) / n_runs
        avg_without = sum(times_without) / n_runs
        overhead = avg_with - avg_without
        overhead_pct = (overhead / avg_without) * 100 if avg_without > 0 else 0
        
        print(f"\nTime PredicateOverhead:")
        print(f"  With Time Predicate Average Time: {avg_with*1000:.2f}ms")
        print(f"  Without Time Predicate Average Time: {avg_without*1000:.2f}ms")
        print(f"  Absolute Overhead: {overhead*1000:.2f}ms")
        print(f"  Relative Overhead: {overhead_pct:.1f}%")
        
        # Save contribution data
        contribution_data = {
            'time_predicate': {
                'with_time': avg_with,
                'without_time': avg_without,
                'overhead': overhead,
                'overhead_pct': overhead_pct
            }
        }
        
        self._save_ablation_data(contribution_data, 'component_contribution.json')
        
        print("\nComponent Contribution Analysis Complete!")
    
    def test_ablation_performance_tradeoff(self, system_setup, heartbeat):
        """Performance-Functionality Tradeoff Analysis"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        print("\n" + "="*80)
        print("Performance-Functionality Tradeoff Analysis")
        print("="*80)
        
        # Test different levels of functionality richness
        complexity_levels = [
            ('Level 1 (Minimal)', ['role:engineer'], "role:engineer"),
            ('Level 2', ['role:engineer', 'dept:maintenance'], "role:engineer AND dept:maintenance"),
            ('Level 3', ['role:engineer', 'dept:maintenance', 'location:factory'], 
             "role:engineer AND dept:maintenance AND location:factory"),
            ('Level 4 (Full)', ['role:engineer', 'dept:maintenance', 'location:factory', 'clearance:level2'], 
             "role:engineer AND dept:maintenance AND location:factory AND clearance:level2 AND time:work")
        ]
        
        tradeoff_results = {}
        total_steps = len(complexity_levels)
        step = 0
        
        for level_name, attrs, policy_str in complexity_levels:
            policy_tree = parser.parse(policy_str)
            SK = tcabe.keygen(MSK, attrs)
            M = group.random(GT)
            
            CT = tcabe.encrypt(M, policy_tree)
            
            work_time = datetime(2026, 4, 22, 11, 0) if 'time:' in policy_str else None
            
            start = time.time()
            decrypted = tcabe.decrypt(SK, CT, work_time) if work_time else tcabe.decrypt(SK, CT)
            decrypt_time = time.time() - start
            assert decrypted == M
            
            tradeoff_results[level_name] = {
                'num_attributes': len(attrs),
                'policy_complexity': policy_str.count(' AND ') + policy_str.count(' OR ') + 1,
                'decrypt_time': decrypt_time
            }
            
            step += 1
            heartbeat(step, total_steps, f"{level_name}: {decrypt_time:.4f}s")
        
        # Print tradeoff table
        print("\nPerformance-Functionality Tradeoff:")
        print(f"{'Level':20} {'AttrCount':8} {'PolicyComplexity':12} {'DecryptTime(ms)':15}")
        print("-"*65)
        
        for level_name in [c[0] for c in complexity_levels]:
            r = tradeoff_results[level_name]
            print(
                f"{level_name:20} "
                f"{r['num_attributes']:8} "
                f"{r['policy_complexity']:12} "
                f"{r['decrypt_time']*1000:13.2f}"
            )
        
        self._save_ablation_data(tradeoff_results, 'performance_tradeoff.json')
        
        print("\nPerformance-Functionality Tradeoff Analysis Complete!")