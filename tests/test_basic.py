"""
Scheme 4: Digital Twin Network Security Framework - Basic Functionality Tests

Proves the system "works"

"""

import pytest
import time
from datetime import datetime
from charm.toolbox.pairinggroup import GT


@pytest.mark.basic
class TestSystemSetup:

    def test_setup_initialization(self, system_setup, heartbeat, memory_monitor, oom_protection):
        s = system_setup
        assert s['PP'] is not None
        assert s['MSK'] is not None
        assert 'g' in s['PP']
        assert 'h' in s['PP']
        assert 'e_gg_alpha' in s['PP']
        assert 'H' in s['PP']
        assert s['group'] is not None
        heartbeat(1, 8, "System initialization verified")
        oom_protection.check_and_protect()

    def test_public_parameters(self, system_setup, heartbeat):
        PP = system_setup['PP']
        assert 'group' in PP
        assert 'g' in PP
        assert 'h' in PP
        assert 'e_gg_alpha' in PP
        assert 'max_attrs' in PP
        heartbeat(2, 8, "Public parameters verified")

    def test_master_key_structure(self, system_setup, heartbeat):
        MSK = system_setup['MSK']
        assert 'alpha' in MSK
        assert 'beta' in MSK
        heartbeat(3, 8, "Master key structure verified")


@pytest.mark.basic
class TestT_CP_ABE:

    def test_key_generation(self, system_setup, heartbeat, oom_protection):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        attrs = ['role:engineer', 'dept:maintenance', 'location:factory']
        SK = tcabe.keygen(MSK, attrs)
        
        assert 'K0' in SK
        assert 'K' in SK
        assert 'r' in SK
        assert 'attributes' in SK
        assert set(attrs) == SK['attributes']
        heartbeat(4, 8, "Key generation verified")
        oom_protection.check_and_protect()

    def test_encrypt_decrypt_success(self, system_setup, heartbeat, memory_monitor):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        attrs = ['role:engineer', 'dept:maintenance']
        SK = tcabe.keygen(MSK, attrs)
        
        policy_tree = parser.parse("role:engineer AND dept:maintenance")
        M = group.random(GT)
        CT = tcabe.encrypt(M, policy_tree)
        
        assert 'C0' in CT
        assert 'C1' in CT
        assert 'leaves' in CT
        
        work_time = datetime(2026, 4, 20, 14, 30)
        decrypted = tcabe.decrypt(SK, CT, work_time)
        assert decrypted == M
        heartbeat(5, 8, "Encrypt-decrypt success verified")
        mem_usage = memory_monitor.get_usage()
        assert mem_usage < 1024

    def test_policy_satisfaction(self, system_setup, heartbeat):
        parser = system_setup['parser']
        
        policy_str = "(role:engineer OR role:admin) AND dept:maintenance"
        tree = parser.parse(policy_str)
        
        assert tree.is_satisfied({'role:engineer', 'dept:maintenance'})
        assert tree.is_satisfied({'role:admin', 'dept:maintenance'})
        assert not tree.is_satisfied({'role:intern', 'dept:maintenance'})
        assert not tree.is_satisfied({'role:engineer'})
        heartbeat(6, 8, "Policy satisfaction verified")

    def test_time_predicate(self, system_setup, heartbeat):
        tcabe = system_setup['tcabe']
        parser = system_setup['parser']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        
        attrs = ['role:engineer', 'dept:maintenance']
        SK = tcabe.keygen(MSK, attrs)
        
        policy_tree = parser.parse("role:engineer AND time:work")
        M = group.random(GT)
        CT = tcabe.encrypt(M, policy_tree)
        
        work_time = datetime(2026, 4, 21, 10, 0)
        decrypted = tcabe.decrypt(SK, CT, work_time)
        assert decrypted == M
        
        night_time = datetime(2026, 4, 21, 22, 0)
        with pytest.raises(ValueError):
            tcabe.decrypt(SK, CT, night_time)
        heartbeat(7, 8, "Time predicate verified")


@pytest.mark.basic
class TestBLSSignature:

    def test_sign_and_verify(self, system_setup, heartbeat):
        bls = system_setup['bls']
        pk, sk = bls.keygen()
        
        message = "test_message_123"
        sigma = bls.sign(sk, message)
        
        assert bls.verify(pk, message, sigma)
        heartbeat(8, 8, "BLS sign and verify passed")

    def test_verify_tampered_message(self, system_setup, heartbeat):
        bls = system_setup['bls']
        pk, sk = bls.keygen()
        
        message = "original_message"
        sigma = bls.sign(sk, message)
        
        assert not bls.verify(pk, "tampered_message", sigma)


@pytest.mark.basic
class TestDeviceCertificate:

    def test_issue_and_verify_certificate(self, system_setup, heartbeat, oom_protection):
        cert_auth = system_setup['cert_auth']
        
        cert = cert_auth.issue_certificate(
            device_id='sensor_001',
            attributes=['type:temperature', 'location:factory']
        )
        
        assert cert_auth.verify_certificate(cert)
        heartbeat(9, 12, "Certificate issue and verify passed")
        oom_protection.check_and_protect()

    def test_forged_certificate_detection(self, system_setup, heartbeat):
        cert_auth = system_setup['cert_auth']
        
        cert = cert_auth.issue_certificate(
            device_id='sensor_001',
            attributes=['type:temperature']
        )
        
        forged1 = dict(cert)
        forged1['device_id'] = 'sensor_forged'
        assert not cert_auth.verify_certificate(forged1)


@pytest.mark.basic
class TestBidirectionalAuth:

    def test_full_auth_flow(self, system_setup, heartbeat, memory_monitor, oom_protection):
        auth = system_setup['auth']
        tcabe = system_setup['tcabe']
        parser = system_setup['parser']
        MSK = system_setup['MSK']
        
        device_id = 'Device_001'
        device_attrs = ['type:sensor', 'location:factory']
        
        auth_request, device_bls_sk = auth.device_auth_init(device_id, device_attrs)
        assert 'nonce_D' in auth_request
        heartbeat(10, 12, "Auth request generated")
        
        policy_T = parser.parse("type:sensor AND location:factory")
        challenge_resp = auth.digital_twin_challenge(auth_request, policy_T)
        assert 'session_id' in challenge_resp
        assert 'enc_challenge' in challenge_resp
        heartbeat(11, 12, "Challenge generated")
        oom_protection.check_and_protect()
        
        device_SK = tcabe.keygen(MSK, device_attrs)
        device_resp = auth.device_response(
            challenge_resp['session_id'],
            challenge_resp,
            device_SK,
            device_bls_sk,
            device_id
        )
        assert 'response' in device_resp
        heartbeat(12, 12, "Device response generated")
        
        verify_result = auth.digital_twin_verify(device_resp)
        assert verify_result['success']
        
        session_info = auth.establish_session(challenge_resp['session_id'])
        assert 'session_key' in session_info
        assert len(session_info['session_key']) == 32
        memory_monitor.get_report()


@pytest.mark.basic
class TestDigitalTwin:

    def test_create_and_get_twin(self, system_setup, heartbeat):
        dt_manager = system_setup['dt_manager']
        
        result = dt_manager.create_digital_twin(
            'factory:sensor_001',
            {'type': 'temperature', 'location': 'factory'}
        )
        assert result['success']
        
        twin = dt_manager.get_digital_twin('factory:sensor_001')
        assert twin is not None
        assert twin['thing_id'] == 'factory:sensor_001'
        heartbeat(13, 16, "Twin created")

    def test_update_twin(self, system_setup, heartbeat):
        dt_manager = system_setup['dt_manager']
        
        dt_manager.create_digital_twin('factory:actuator_001', {'type': 'valve'})
        result = dt_manager.update_digital_twin(
            'factory:actuator_001',
            {'status': 'active', 'value': 42.5}
        )
        assert result['success']
        
        twin = dt_manager.get_digital_twin('factory:actuator_001')
        assert twin['properties']['status'] == 'active'
        heartbeat(14, 16, "Twin updated")

    def test_send_command(self, system_setup, heartbeat, oom_protection):
        dt_manager = system_setup['dt_manager']
        
        dt_manager.create_digital_twin('factory:sensor_002', {'type': 'pressure'})
        result = dt_manager.send_command(
            'factory:sensor_002',
            {'action': 'read_sensor', 'params': {'type': 'pressure'}}
        )
        assert result['success']
        assert result['response']['status'] == 'success'
        heartbeat(15, 16, "Command sent")
        oom_protection.check_and_protect()

    def test_statistics(self, system_setup, heartbeat):
        dt_manager = system_setup['dt_manager']
        
        stats = dt_manager.get_statistics()
        assert 'total_twins' in stats
        assert 'online_twins' in stats
        assert 'total_commands' in stats
        assert 'total_threat_events' in stats
        heartbeat(16, 16, "Statistics retrieved")


@pytest.mark.basic
class TestSubprocessWorker:

    def test_simple_task_execution(self, system_setup, heartbeat):
        worker = system_setup['worker']
        
        def add(x, y):
            return x + y
        
        result = worker.execute_with_isolation(add, 10, 20)
        assert result['success']
        assert result['result'] == 30
        heartbeat(17, 19, "Simple task executed")

    def test_exception_isolation(self, system_setup, heartbeat):
        worker = system_setup['worker']
        
        def failing():
            raise ValueError("Intentional failure")
        
        result = worker.execute_with_isolation(failing)
        assert not result['success']
        assert 'ValueError' in result['error']
        heartbeat(18, 19, "Exception isolated")


@pytest.mark.basic
class TestDiffusionModel:

    def test_model_initialization(self, heartbeat, memory_monitor, oom_protection):
        torch = pytest.importorskip("torch")
        from src.diffusion import ThreatDiffusionModel
        
        model = ThreatDiffusionModel(
            vocab_size=50, embed_dim=32, condition_dim=16,
            num_train_timesteps=10, device='cpu'
        )
        
        assert model is not None
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        heartbeat(19, 19, "Model initialized")
        oom_protection.check_and_protect()
        memory_monitor.get_usage()

    def test_anomaly_score(self, heartbeat):
        torch = pytest.importorskip("torch")
        from src.diffusion import ThreatDiffusionModel
        
        model = ThreatDiffusionModel(
            vocab_size=50, embed_dim=32, condition_dim=16,
            num_train_timesteps=10, device='cpu'
        )
        
        normal_request = {'attrs': [10, 20, 30]}
        normal_context = {'time_anomaly': False, 'behavior_anomaly': False}
        normal_score = model.anomaly_score(normal_request, normal_context)
        assert 0.0 <= normal_score <= 1.0
        
        suspicious_request = {'attrs': [45, 46, 47], 'suspicious_attrs': True}
        suspicious_context = {'time_anomaly': True, 'behavior_anomaly': True}
        suspicious_score = model.anomaly_score(suspicious_request, suspicious_context)
        assert 0.0 <= suspicious_score <= 1.0

    def test_adaptive_policy(self, heartbeat):
        torch = pytest.importorskip("torch")
        from src.diffusion import ThreatDiffusionModel
        
        model = ThreatDiffusionModel(
            vocab_size=50, embed_dim=32, condition_dim=16,
            num_train_timesteps=10, device='cpu'
        )
        
        base_policy = "role:engineer"
        
        low_policy = model.adaptive_policy_update(0.1, base_policy)
        assert low_policy == base_policy
        
        med_policy = model.adaptive_policy_update(0.5, base_policy)
        assert "dept:engineering" in med_policy
        
        high_policy = model.adaptive_policy_update(0.9, base_policy)
        assert "mfa:true" in high_policy
        heartbeat(20, 20, "Adaptive policy verified")
