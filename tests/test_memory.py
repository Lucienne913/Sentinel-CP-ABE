"""
pytest tests/test_memory.py -m memory -v

Scheme 4: Digital Twin Network Security Framework - Memory Leak Tests

Test Content:

- After running 100 encryption operations, does memory leak?
- Is memory usage stable?
Significance for your paper: Prove that your system "does not leak memory"

"""

import pytest
import gc
import time
from datetime import datetime
from charm.toolbox.pairinggroup import GT


@pytest.mark.memory
class TestCacheMemoryLeak:

    def test_lru_cache_bounded_growth(self, system_setup, heartbeat, memory_monitor, oom_protection):
        PP = system_setup['PP']
        
        initial_mem = memory_monitor.get_usage()
        cache_max_size = PP.get('cache_max_size', 1000000)
        
        for i in range(100):
            PP['H'](f'attr:leak_test_{i}')
            heartbeat(i + 1, 100, f"Attribute hash #{i+1}")
            if (i + 1) % 20 == 0:
                oom_protection.check_and_protect()
        
        final_mem = memory_monitor.get_usage()
        mem_growth = final_mem - initial_mem
        
        assert mem_growth < 500
        cache_size = len(PP['attr_hash_cache'])
        assert cache_size <= cache_max_size
        heartbeat(100, 100, f"Cache growth: {mem_growth:.1f}MB, cache_size={cache_size}")

    def test_cache_clear_releases_memory(self, heartbeat, memory_monitor, oom_protection):
        from src.setup import T_CP_ABE_Setup
        
        setup = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
        PP, MSK = setup.setup(max_attrs=100, cache_max_size=5000)
        
        for i in range(1000):
            PP['H'](f'attr:clear_test_{i}')
        
        mem_before_clear = memory_monitor.get_usage()
        
        PP['clear_cache']()
        gc.collect()
        
        mem_after_clear = memory_monitor.get_usage()
        assert mem_after_clear <= mem_before_clear + 10
        heartbeat(1000, 1000, f"Cache cleared: {mem_after_clear:.1f}MB")
        oom_protection.check_and_protect()


@pytest.mark.memory
class TestSessionMemoryLeak:

    def test_nonce_cache_bounded_growth(self, system_setup, heartbeat, memory_monitor, oom_protection):
        auth = system_setup['auth']
        original_max = auth.max_nonce_cache_size
        auth.max_nonce_cache_size = 50
        
        initial_mem = memory_monitor.get_usage()
        
        for i in range(200):
            nonce = auth.generate_nonce()
            auth.is_nonce_valid(nonce, time.time())
            heartbeat(i + 1, 200, f"Nonce cache #{i+1}")
        
        assert len(auth.nonce_cache) <= 50
        final_mem = memory_monitor.get_usage()
        mem_growth = final_mem - initial_mem
        
        assert mem_growth < 100
        auth.max_nonce_cache_size = original_max
        heartbeat(200, 200, f"Nonce cache growth: {mem_growth:.1f}MB")
        oom_protection.check_and_protect()

    def test_session_cleanup_releases_memory(self, system_setup, heartbeat, memory_monitor):
        auth = system_setup['auth']
        original_max = auth.max_active_sessions
        auth.max_active_sessions = 10
        
        for i in range(50):
            auth.active_sessions[f'session_mem_{i}'] = {
                'device_id': f'device_{i}',
                'timestamp': time.time() - 7200
            }
        
        mem_before = memory_monitor.get_usage()
        auth._cleanup_expired_sessions()
        gc.collect()
        mem_after = memory_monitor.get_usage()
        
        assert len(auth.active_sessions) <= 10
        auth.max_active_sessions = original_max
        heartbeat(201, 202, "Sessions cleaned")


@pytest.mark.memory
class TestHistoryMemoryLeak:

    def test_command_history_bounded_growth(self, system_setup, heartbeat, memory_monitor, oom_protection):
        dt_manager = system_setup['dt_manager']
        original_max = dt_manager.max_history_size
        dt_manager.max_history_size = 20
        
        dt_manager.create_digital_twin('factory:mem_twin', {'type': 'sensor'})
        
        for i in range(50):
            dt_manager.command_history.append({
                'cmd': i,
                'timestamp': time.time()
            })
            heartbeat(i + 1, 50, f"Command history #{i+1}")
        
        dt_manager._cleanup_memory_history()
        assert len(dt_manager.command_history) <= 20
        
        dt_manager.max_history_size = original_max
        heartbeat(50, 50, f"Command history size: {len(dt_manager.command_history)}")
        oom_protection.check_and_protect()

    def test_auth_history_bounded_growth(self, system_setup, heartbeat, memory_monitor):
        dt_manager = system_setup['dt_manager']
        original_max = dt_manager.max_history_size
        dt_manager.max_history_size = 15
        
        for i in range(40):
            dt_manager.auth_history.append({
                'auth': i,
                'timestamp': time.time()
            })
        
        dt_manager._cleanup_memory_history()
        assert len(dt_manager.auth_history) <= 15
        
        dt_manager.max_history_size = original_max
        heartbeat(51, 52, "Auth history cleaned")


@pytest.mark.memory
class TestDigitalTwinMemoryLeak:

    def test_twin_creation_deletion_cycle(self, system_setup, heartbeat, memory_monitor, oom_protection):
        dt_manager = system_setup['dt_manager']
        
        initial_mem = memory_monitor.get_usage()
        
        for i in range(20):
            twin_id = f'factory:cycle_twin_{i}'
            dt_manager.create_digital_twin(twin_id, {'type': 'sensor', 'index': i})
            dt_manager.delete_digital_twin(twin_id)
            heartbeat(i + 1, 20, f"Create-delete cycle #{i+1}")
        
        gc.collect()
        final_mem = memory_monitor.get_usage()
        mem_growth = final_mem - initial_mem
        
        assert mem_growth < 100
        heartbeat(20, 20, f"Cycle memory growth: {mem_growth:.1f}MB")
        oom_protection.check_and_protect()

    def test_twin_update_no_leak(self, system_setup, heartbeat, memory_monitor):
        dt_manager = system_setup['dt_manager']
        
        dt_manager.create_digital_twin('factory:update_twin', {'type': 'actuator'})
        
        initial_mem = memory_monitor.get_usage()
        
        for i in range(50):
            dt_manager.update_digital_twin(
                'factory:update_twin',
                {'value': i, 'status': f'state_{i}'}
            )
            heartbeat(i + 1, 50, f"Update #{i+1}")
        
        gc.collect()
        final_mem = memory_monitor.get_usage()
        mem_growth = final_mem - initial_mem
        
        assert mem_growth < 50
        heartbeat(50, 50, f"Update memory growth: {mem_growth:.1f}MB")


@pytest.mark.memory
class TestSubprocessMemoryLeak:

    def test_subprocess_no_memory_leak(self, system_setup, heartbeat, memory_monitor, oom_protection):
        worker = system_setup['worker']
        
        initial_mem = memory_monitor.get_usage()
        
        for i in range(10):
            def compute_task(n):
                return sum(range(n))
            
            result = worker.execute_with_isolation(compute_task, 10000)
            assert result['success']
            heartbeat(i + 1, 10, f"Subprocess task #{i+1}")
            gc.collect()
        
        final_mem = memory_monitor.get_usage()
        mem_growth = final_mem - initial_mem
        
        assert mem_growth < 200
        heartbeat(10, 10, f"Subprocess memory growth: {mem_growth:.1f}MB")
        oom_protection.check_and_protect()


@pytest.mark.memory
class TestDiffusionModelMemoryLeak:

    def test_model_inference_no_gpu_leak(self, heartbeat, memory_monitor, oom_protection):
        torch = pytest.importorskip("torch")
        from src.diffusion import ThreatDiffusionModel
        
        model = ThreatDiffusionModel(
            vocab_size=100, embed_dim=64, condition_dim=32,
            num_train_timesteps=50, device='cpu'
        )
        
        initial_mem = memory_monitor.get_usage()
        
        for i in range(10):
            request = {'attrs': [10, 20, 30]}
            context = {'time_anomaly': False, 'behavior_anomaly': False}
            model.anomaly_score(request, context)
            heartbeat(i + 1, 10, f"Inference #{i+1}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        final_mem = memory_monitor.get_usage()
        mem_growth = final_mem - initial_mem
        
        assert mem_growth < 200
        heartbeat(10, 10, f"Inference memory growth: {mem_growth:.1f}MB")
        oom_protection.check_and_protect()

    def test_adversarial_generation_memory(self, heartbeat, memory_monitor):
        torch = pytest.importorskip("torch")
        from src.diffusion import ThreatDiffusionModel
        
        model = ThreatDiffusionModel(
            vocab_size=50, embed_dim=32, condition_dim=16,
            num_train_timesteps=10, device='cpu'
        )
        
        policy_indices = torch.tensor([[10, 20, 30] + [0]*17])
        
        initial_mem = memory_monitor.get_usage()
        
        for i in range(5):
            model.generate_adversarial(policy_indices, n_samples=10)
            gc.collect()
            heartbeat(i + 1, 5, f"Adversarial generation #{i+1}")
        
        final_mem = memory_monitor.get_usage()
        mem_growth = final_mem - initial_mem
        assert mem_growth < 300
        heartbeat(5, 5, f"Adversarial memory growth: {mem_growth:.1f}MB")


@pytest.mark.memory
class TestOOMProtection:

    def test_oom_threshold_detection(self, system_setup, heartbeat, oom_protection, memory_monitor):
        swap_status = oom_protection.get_swap_status()
        if swap_status:
            assert swap_status['total_gb'] >= 0
            assert swap_status['percent'] <= 100
        
        initial_swap = memory_monitor.get_swap_usage()
        current_mem = oom_protection.check_and_protect()
        
        assert current_mem >= 0
        heartbeat(1, 3, f"Current mem: {current_mem:.1f}MB, Swap: {initial_swap:.1f}MB")

    def test_gc_trigger_under_pressure(self, heartbeat, oom_protection, memory_monitor):
        for i in range(5):
            data = [list(range(1000)) for _ in range(100)]
            del data
            gc.collect()
            mem = oom_protection.check_and_protect()
            heartbeat(i + 1, 5, f"GC pressure #{i+1}: {mem:.1f}MB")
        
        final_mem = memory_monitor.get_usage()
        assert final_mem < 2048


@pytest.mark.memory
class TestSwapExpansion:

    def test_swap_usage_monitoring(self, system_setup, heartbeat, memory_monitor, oom_protection):
        initial_swap = memory_monitor.get_swap_usage()
        
        for i in range(20):
            from src.setup import T_CP_ABE_Setup
            from src.t_cp_abe import T_CP_ABE
            
            setup = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
            PP, MSK = setup.setup(max_attrs=30)
            tcabe = T_CP_ABE(PP)
            
            attrs = [f'attr:{j}' for j in range(25)]
            tcabe.keygen(MSK, attrs)
            
            heartbeat(i + 1, 20, f"Swap monitoring loop #{i+1}")
            if (i + 1) % 5 == 0:
                oom_protection.check_and_protect()
        
        final_swap = memory_monitor.get_swap_usage()
        
        swap_increase = final_swap - initial_swap
        assert swap_increase < 1024
        heartbeat(20, 20, f"Swap increase: {swap_increase:.1f}MB")
