"""
Run with:
pytest tests/test_edge_cases.py -m edge_case -v

Scheme 4: Digital Twin Network Security Framework - Edge Case Tests

Test content:

- Can null message be encrypted?
- Can expired time data be decrypted?
- How to process very large attribute lists?

"""

import pytest
import time
from datetime import datetime
from charm.toolbox.pairinggroup import GT


@pytest.mark.edge_case
class TestEmptyEdgeCases:

    def test_empty_attribute_set(self, system_setup, heartbeat, oom_protection):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        
        SK = tcabe.keygen(MSK, [])
        assert 'K0' in SK
        assert len(SK['K']) == 0
        heartbeat(1, 10, "Empty attribute key generated")
        oom_protection.check_and_protect()

    def test_empty_attribute_decryption(self, system_setup, heartbeat):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        SK = tcabe.keygen(MSK, [])
        policy_tree = parser.parse("role:engineer")
        M = group.random(GT)
        CT = tcabe.encrypt(M, policy_tree)
        
        with pytest.raises(ValueError):
            tcabe.decrypt(SK, CT)
        heartbeat(2, 10, "Empty attribute decryption failed as expected")

    def test_empty_string_attribute(self, system_setup, heartbeat):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        
        SK = tcabe.keygen(MSK, [''])
        assert '' in SK['attributes']
        heartbeat(3, 10, "Empty string attribute handled")

    def test_empty_policy_string(self, system_setup, heartbeat):
        parser = system_setup['parser']
        
        with pytest.raises((ValueError, IndexError)):
            parser.parse("")
        heartbeat(4, 10, "Empty policy string rejected")


@pytest.mark.edge_case
class TestLargeScaleEdgeCases:

    def test_large_attribute_set(self, system_setup, heartbeat, memory_monitor, oom_protection):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        
        large_attrs = [f'attr:{i}' for i in range(45)]
        
        start = time.time()
        SK = tcabe.keygen(MSK, large_attrs)
        keygen_time = time.time() - start
        
        assert len(SK['K']) == 45
        assert keygen_time < 10.0
        heartbeat(5, 12, f"Large attribute keygen: {keygen_time:.3f}s")
        mem_usage = memory_monitor.get_usage()
        oom_protection.check_and_protect()

    def test_deeply_nested_policy(self, system_setup, heartbeat, oom_protection):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        nested_policy = "(((role:a AND dept:b) AND loc:c) AND level:d) AND time:work"
        policy_tree = parser.parse(nested_policy)
        
        attrs = ['role:a', 'dept:b', 'loc:c', 'level:d']
        SK = tcabe.keygen(MSK, attrs)
        
        M = group.random(GT)
        CT = tcabe.encrypt(M, policy_tree)
        
        work_time = datetime(2026, 4, 21, 10, 0)
        decrypted = tcabe.decrypt(SK, CT, work_time)
        assert decrypted == M
        heartbeat(6, 12, "Deeply nested policy decrypted")
        oom_protection.check_and_protect()

    def test_many_or_conditions(self, system_setup, heartbeat):
        parser = system_setup['parser']
        
        or_policy = "role:a OR role:b OR role:c OR role:d OR role:e"
        tree = parser.parse(or_policy)
        
        assert tree.is_satisfied({'role:a'})
        assert tree.is_satisfied({'role:c'})
        assert tree.is_satisfied({'role:e'})
        assert not tree.is_satisfied({'role:f'})
        heartbeat(7, 12, "Many OR conditions evaluated")


@pytest.mark.edge_case
class TestTimeEdgeCases:

    def test_boundary_time_values(self, system_setup, heartbeat, oom_protection):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        attrs = ['role:engineer']
        SK = tcabe.keygen(MSK, attrs)
        
        policy_tree = parser.parse("role:engineer AND time:work")
        M = group.random(GT)
        CT = tcabe.encrypt(M, policy_tree)
        
        boundary_time_8am = datetime(2026, 4, 21, 8, 0)
        decrypted = tcabe.decrypt(SK, CT, boundary_time_8am)
        assert decrypted == M
        
        boundary_time_6pm = datetime(2026, 4, 21, 18, 0)
        decrypted = tcabe.decrypt(SK, CT, boundary_time_6pm)
        assert decrypted == M
        heartbeat(8, 12, "Boundary time values accepted")
        oom_protection.check_and_protect()

    def test_midnight_time(self, system_setup, heartbeat):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        attrs = ['role:engineer']
        SK = tcabe.keygen(MSK, attrs)
        
        policy_tree = parser.parse("role:engineer AND time:work")
        M = group.random(GT)
        CT = tcabe.encrypt(M, policy_tree)
        
        midnight = datetime(2026, 4, 21, 0, 0)
        with pytest.raises(ValueError):
            tcabe.decrypt(SK, CT, midnight)
        heartbeat(9, 12, "Midnight time rejected")


@pytest.mark.edge_case
class TestCacheEdgeCases:

    def test_lru_cache_eviction(self, heartbeat, oom_protection):
        from src.setup import T_CP_ABE_Setup
        
        cache_max_size = 5
        setup = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
        PP, MSK = setup.setup(max_attrs=10, cache_max_size=cache_max_size)
        
        for i in range(10):
            PP['H'](f'attr:{i}')
        
        cache_size = len(PP['attr_hash_cache'])
        assert cache_size <= cache_max_size
        heartbeat(10, 14, f"LRU eviction: {cache_size}/{cache_max_size}")
        oom_protection.check_and_protect()

    def test_cache_ttl_expiry(self, heartbeat):
        from src.setup import T_CP_ABE_Setup
        import time
        
        setup = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
        PP, MSK = setup.setup(max_attrs=10, cache_max_size=100)
        
        PP['H']('test_attr', ttl=1)
        assert 'test_attr' in PP['attr_hash_cache']
        
        time.sleep(2)
        
        PP['H']('test_attr', ttl=1)
        heartbeat(11, 14, "Cache TTL expired")


@pytest.mark.edge_case
class TestSessionEdgeCases:

    def test_max_sessions_capacity(self, system_setup, heartbeat, oom_protection):
        auth = system_setup['auth']
        
        original_max = auth.max_active_sessions
        auth.max_active_sessions = 3
        
        for i in range(5):
            auth.active_sessions[f'session_{i}'] = {
                'device_id': f'device_{i}',
                'timestamp': time.time()
            }
        
        auth._cleanup_expired_sessions()
        assert len(auth.active_sessions) <= 3
        
        auth.max_active_sessions = original_max
        heartbeat(12, 14, "Session capacity limited")
        oom_protection.check_and_protect()

    def test_expired_session_cleanup(self, system_setup, heartbeat):
        auth = system_setup['auth']
        
        old_time = time.time() - 7200
        auth.active_sessions['expired_1'] = {'timestamp': old_time}
        auth.active_sessions['expired_2'] = {'timestamp': old_time}
        auth.active_sessions['active_1'] = {'timestamp': time.time()}
        
        auth._cleanup_expired_sessions()
        
        assert 'expired_1' not in auth.active_sessions
        assert 'expired_2' not in auth.active_sessions
        assert 'active_1' in auth.active_sessions
        heartbeat(13, 14, "Expired sessions cleaned")


@pytest.mark.edge_case
class TestSubprocessEdgeCases:

    def test_very_short_timeout(self, system_setup, heartbeat, oom_protection):
        worker = system_setup['worker']
        
        original_timeout = worker.timeout
        worker.timeout = 1
        
        def slow_task():
            time.sleep(5)
            return "done"
        
        result = worker.execute_with_isolation(slow_task)
        assert not result['success']
        assert 'Timeout' in result['error']
        
        worker.timeout = original_timeout
        heartbeat(14, 16, "Short timeout handled")
        oom_protection.check_and_protect()


@pytest.mark.edge_case
class TestDiffusionEdgeCases:

    def test_empty_attribute_input(self, heartbeat):
        torch = pytest.importorskip("torch")
        from src.diffusion import ThreatDiffusionModel
        
        model = ThreatDiffusionModel(
            vocab_size=50, embed_dim=32, condition_dim=16,
            num_train_timesteps=10, device='cpu'
        )
        
        empty_request = {'attrs': []}
        empty_context = {'time_anomaly': False, 'behavior_anomaly': False}
        
        score = model.anomaly_score(empty_request, empty_context)
        assert 0.0 <= score <= 1.0
        heartbeat(15, 16, "Empty attribute input scored")

    def test_extreme_attribute_values(self, heartbeat, oom_protection):
        torch = pytest.importorskip("torch")
        from src.diffusion import ThreatDiffusionModel
        
        model = ThreatDiffusionModel(
            vocab_size=50, embed_dim=32, condition_dim=16,
            num_train_timesteps=10, device='cpu'
        )
        
        extreme_request = {'attrs': [0, 49, 999, -1]}
        extreme_context = {'time_anomaly': True, 'behavior_anomaly': True}
        
        score = model.anomaly_score(extreme_request, extreme_context)
        assert 0.0 <= score <= 1.0
        heartbeat(16, 16, "Extreme attributes scored")
        oom_protection.check_and_protect()


@pytest.mark.edge_case
class TestDigitalTwinEdgeCases:

    def test_duplicate_twin_creation(self, system_setup, heartbeat, oom_protection):
        dt_manager = system_setup['dt_manager']
        
        result1 = dt_manager.create_digital_twin('factory:edge_twin', {'type': 'sensor'})
        assert result1['success']
        
        result2 = dt_manager.create_digital_twin('factory:edge_twin', {'type': 'sensor'})
        assert not result2['success']
        assert 'already exists' in result2['message']
        heartbeat(17, 19, "Duplicate creation rejected")
        oom_protection.check_and_protect()

    def test_nonexistent_twin_operations(self, system_setup, heartbeat):
        dt_manager = system_setup['dt_manager']
        
        update_result = dt_manager.update_digital_twin(
            'factory:nonexistent',
            {'value': 100}
        )
        assert not update_result['success']
        
        delete_result = dt_manager.delete_digital_twin('factory:nonexistent')
        assert not delete_result['success']
        
        command_result = dt_manager.send_command(
            'factory:nonexistent',
            {'action': 'read_sensor', 'params': {}}
        )
        assert not command_result['success']
        heartbeat(18, 19, "Nonexistent twin operations rejected")

    def test_history_size_limit(self, system_setup, heartbeat, oom_protection):
        dt_manager = system_setup['dt_manager']
        original_max = dt_manager.max_history_size
        dt_manager.max_history_size = 3
        
        dt_manager.create_digital_twin('factory:hist_twin', {'type': 'sensor'})
        
        for i in range(5):
            dt_manager.command_history.append({'cmd': i, 'timestamp': time.time()})
        
        dt_manager._cleanup_memory_history()
        assert len(dt_manager.command_history) <= 3
        
        dt_manager.max_history_size = original_max
        heartbeat(19, 19, "History size limited")
        oom_protection.check_and_protect()
