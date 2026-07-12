"""
Scheme 4：Digital Twin Network Security Framework - Performance Test
"""

import pytest
import time
from datetime import datetime
from charm.toolbox.pairinggroup import GT


@pytest.mark.performance
class TestEncryptionPerformance:

    def test_encryption_time_small_policy(self, system_setup, heartbeat, memory_monitor, oom_protection):
        tcabe = system_setup['tcabe']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        policy_tree = parser.parse("role:engineer")
        M = group.random(GT)
        
        start = time.time()
        CT = tcabe.encrypt(M, policy_tree)
        encrypt_time = time.time() - start
        
        assert encrypt_time < 1.0
        assert CT is not None
        heartbeat(1, 8, f"Small policy encryption: {encrypt_time:.4f}s")
        memory_monitor.get_usage()
        oom_protection.check_and_protect()

    def test_encryption_time_medium_policy(self, system_setup, heartbeat, memory_monitor):
        tcabe = system_setup['tcabe']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        policy_tree = parser.parse("role:engineer AND dept:maintenance AND location:factory")
        M = group.random(GT)
        
        start = time.time()
        CT = tcabe.encrypt(M, policy_tree)
        encrypt_time = time.time() - start
        
        assert encrypt_time < 2.0
        heartbeat(2, 8, f"Medium policy encryption: {encrypt_time:.4f}s")
        memory_monitor.get_usage()

    def test_encryption_time_complex_policy(self, system_setup, heartbeat, oom_protection):
        tcabe = system_setup['tcabe']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        policy_tree = parser.parse("(role:engineer OR role:admin) AND dept:maintenance AND time:work")
        M = group.random(GT)
        
        start = time.time()
        CT = tcabe.encrypt(M, policy_tree)
        encrypt_time = time.time() - start
        
        assert encrypt_time < 3.0
        heartbeat(3, 8, f"Complex policy encryption: {encrypt_time:.4f}s")
        oom_protection.check_and_protect()


@pytest.mark.performance
class TestDecryptionPerformance:

    def test_decryption_time_single_attr(self, system_setup, heartbeat, memory_monitor):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        attrs = ['role:engineer']
        SK = tcabe.keygen(MSK, attrs)
        policy_tree = parser.parse("role:engineer")
        M = group.random(GT)
        CT = tcabe.encrypt(M, policy_tree)
        
        start = time.time()
        decrypted = tcabe.decrypt(SK, CT)
        decrypt_time = time.time() - start
        
        assert decrypted == M
        assert decrypt_time < 1.0
        heartbeat(4, 8, f"Single attr decryption: {decrypt_time:.4f}s")
        memory_monitor.get_usage()

    def test_decryption_time_multiple_attrs(self, system_setup, heartbeat, oom_protection):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        attrs = ['role:engineer', 'dept:maintenance', 'location:factory', 'clearance:level3']
        SK = tcabe.keygen(MSK, attrs)
        policy_tree = parser.parse("role:engineer AND dept:maintenance AND location:factory AND clearance:level3")
        M = group.random(GT)
        CT = tcabe.encrypt(M, policy_tree)
        
        start = time.time()
        decrypted = tcabe.decrypt(SK, CT)
        decrypt_time = time.time() - start
        
        assert decrypted == M
        assert decrypt_time < 2.0
        heartbeat(5, 8, f"Multiple attrs decryption: {decrypt_time:.4f}s")
        oom_protection.check_and_protect()


@pytest.mark.performance
class TestBatchOperations:

    def test_batch_encryption_throughput(self, system_setup, heartbeat, memory_monitor, oom_protection):
        tcabe = system_setup['tcabe']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        policy_tree = parser.parse("role:engineer AND dept:maintenance")
        batch_size = 10
        
        start = time.time()
        ciphertexts = []
        for i in range(batch_size):
            M = group.random(GT)
            CT = tcabe.encrypt(M, policy_tree)
            ciphertexts.append(CT)
            heartbeat(6 + i, 6 + batch_size + 2, f"Batch encryption {i+1}/{batch_size}")
        
        total_time = time.time() - start
        throughput = batch_size / total_time
        
        assert len(ciphertexts) == batch_size
        assert throughput > 1.0
        heartbeat(6 + batch_size, 6 + batch_size + 2, f"Batch encryption throughput: {throughput:.2f} ops/s")
        memory_monitor.get_usage()
        oom_protection.check_and_protect()

    def test_batch_decryption_throughput(self, system_setup, heartbeat, memory_monitor):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        attrs = ['role:engineer', 'dept:maintenance']
        SK = tcabe.keygen(MSK, attrs)
        policy_tree = parser.parse("role:engineer AND dept:maintenance")
        
        batch_size = 5
        messages = []
        ciphertexts = []
        for i in range(batch_size):
            M = group.random(GT)
            messages.append(M)
            CT = tcabe.encrypt(M, policy_tree)
            ciphertexts.append(CT)
        
        start = time.time()
        decrypted = []
        for i, CT in enumerate(ciphertexts):
            M = tcabe.decrypt(SK, CT)
            decrypted.append(M)
            heartbeat(16 + i, 16 + batch_size + 2, f"Batch decryption {i+1}/{batch_size}")
        
        total_time = time.time() - start
        throughput = batch_size / total_time
        
        assert all(d == m for d, m in zip(decrypted, messages))
        heartbeat(16 + batch_size, 16 + batch_size + 2, f"Batch decryption throughput: {throughput:.2f} ops/s")
        memory_monitor.get_usage()


@pytest.mark.performance
class TestSignaturePerformance:

    def test_signing_latency(self, system_setup, heartbeat, oom_protection):
        bls = system_setup['bls']
        pk, sk = bls.keygen()
        
        latencies = []
        for i in range(20):
            message = f"test_message_{i}"
            start = time.time()
            sigma = bls.sign(sk, message)
            latency = time.time() - start
            latencies.append(latency)
            heartbeat(21 + i, 21 + 20 + 1, f"Signing latency #{i+1}")
        
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 0.5
        oom_protection.check_and_protect()

    def test_verification_latency(self, system_setup, heartbeat):
        bls = system_setup['bls']
        pk, sk = bls.keygen()
        
        message = "test_message"
        sigma = bls.sign(sk, message)
        
        latencies = []
        for _ in range(20):
            start = time.time()
            valid = bls.verify(pk, message, sigma)
            latency = time.time() - start
            latencies.append(latency)
            assert valid
        
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 0.5
        heartbeat(41, 42, f"Verification latency: {avg_latency*1000:.2f}ms")


@pytest.mark.performance
class TestDigitalTwinPerformance:

    def test_twin_creation_latency(self, system_setup, heartbeat, memory_monitor, oom_protection):
        dt_manager = system_setup['dt_manager']
        
        latencies = []
        for i in range(10):
            start = time.time()
            result = dt_manager.create_digital_twin(
                f'factory:perf_twin_{i}',
                {'type': 'sensor', 'index': i}
            )
            latency = time.time() - start
            latencies.append(latency)
            assert result['success']
            heartbeat(42 + i, 42 + 10 + 1, f"Twin creation #{i+1}")
        
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 0.1
        memory_monitor.get_usage()
        oom_protection.check_and_protect()

    def test_command_execution_latency(self, system_setup, heartbeat):
        dt_manager = system_setup['dt_manager']
        
        dt_manager.create_digital_twin('factory:perf_device', {'type': 'actuator'})
        
        latencies = []
        for i in range(10):
            start = time.time()
            result = dt_manager.send_command(
                'factory:perf_device',
                {'action': 'read_sensor', 'params': {'type': 'temp'}}
            )
            latency = time.time() - start
            latencies.append(latency)
            assert result['success']
            heartbeat(52 + i, 52 + 10 + 1, f"Command execution #{i+1}")
        
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 0.1
        heartbeat(62, 63, f"Command latency: {avg_latency*1000:.2f}ms")


@pytest.mark.performance
class TestSubprocessPerformance:

    def test_subprocess_overhead(self, system_setup, heartbeat, memory_monitor, oom_protection):
        worker = system_setup['worker']
        
        def simple_task(x):
            return x * 2
        
        start = time.time()
        result = worker.execute_with_isolation(simple_task, 42)
        overhead = time.time() - start
        
        assert result['success']
        assert result['result'] == 84
        assert overhead < 5.0
        heartbeat(63, 64, f"Subprocess overhead: {overhead:.3f}s")
        memory_monitor.get_usage()
        swap_usage = memory_monitor.get_swap_usage()
        oom_protection.check_and_protect()


@pytest.mark.performance
class TestDiffusionModelPerformance:

    def test_model_inference_latency(self, heartbeat, memory_monitor, oom_protection):
        torch = pytest.importorskip("torch")
        from src.diffusion import ThreatDiffusionModel
        
        model = ThreatDiffusionModel(
            vocab_size=100, embed_dim=64, condition_dim=32,
            num_train_timesteps=50, device='cpu'
        )
        
        start = time.time()
        normal_request = {'attrs': [10, 20, 30]}
        normal_context = {'time_anomaly': False, 'behavior_anomaly': False}
        score = model.anomaly_score(normal_request, normal_context)
        latency = time.time() - start
        
        assert 0.0 <= score <= 1.0
        assert latency < 2.0
        heartbeat(64, 66, f"Inference latency: {latency*1000:.2f}ms")
        memory_monitor.get_usage()
        oom_protection.check_and_protect()

    def test_adversarial_generation_time(self, heartbeat, memory_monitor):
        torch = pytest.importorskip("torch")
        from src.diffusion import ThreatDiffusionModel
        
        model = ThreatDiffusionModel(
            vocab_size=50, embed_dim=32, condition_dim=16,
            num_train_timesteps=10, device='cpu'
        )
        
        policy_indices = torch.tensor([[10, 20, 30] + [0]*17])
        
        start = time.time()
        adversarial = model.generate_adversarial(policy_indices, n_samples=5)
        gen_time = time.time() - start
        
        assert adversarial.shape[0] == 5
        assert gen_time < 5.0
        heartbeat(66, 67, f"Adversarial generation time: {gen_time:.3f}s")
        memory_monitor.get_usage()


@pytest.mark.performance
class TestSwapMonitoring:

    def test_swap_usage_during_operations(self, system_setup, heartbeat, memory_monitor, oom_protection):
        initial_swap = memory_monitor.get_swap_usage()
        
        for i in range(5):
            from src.setup import T_CP_ABE_Setup
            from src.t_cp_abe import T_CP_ABE
            
            setup = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
            PP, MSK = setup.setup(max_attrs=30)
            tcabe = T_CP_ABE(PP)
            
            attrs = [f'attr:{j}' for j in range(25)]
            tcabe.keygen(MSK, attrs)
            heartbeat(i + 1, 5, f"Swap monitoring loop #{i+1}")
            oom_protection.check_and_protect()
        
        final_swap = memory_monitor.get_swap_usage()
        swap_increase = final_swap - initial_swap
        
        swap_status = oom_protection.get_swap_status()
        if swap_status:
            assert swap_status['percent'] < 90
        heartbeat(5, 5, f"Swap increase: {swap_increase:.1f}MB")
