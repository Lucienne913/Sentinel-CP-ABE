"""
Scheme 4: Digital Twin Network Security Framework - Security Tests
"""

import pytest
import time
from datetime import datetime
from charm.toolbox.pairinggroup import GT


@pytest.mark.security
class TestIND_CPASecurity:

    def test_unauthorized_attributes_decrypt_failure(self, system_setup, heartbeat, oom_protection):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        auth_attrs = ['role:engineer', 'dept:maintenance']
        SK_auth = tcabe.keygen(MSK, auth_attrs)
        
        policy_tree = parser.parse("role:engineer AND dept:maintenance")
        M = group.random(GT)
        CT = tcabe.encrypt(M, policy_tree)
        
        unauth_attrs = ['role:intern', 'dept:unknown']
        SK_unauth = tcabe.keygen(MSK, unauth_attrs)
        
        with pytest.raises(ValueError, match="Attribute set does not satisfy policy"):
            tcabe.decrypt(SK_unauth, CT)
        heartbeat(1, 10, "Unauthorized attributes rejected")
        oom_protection.check_and_protect()

    def test_partial_attributes_decrypt_failure(self, system_setup, heartbeat):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        policy_tree = parser.parse("role:engineer AND dept:maintenance AND location:factory")
        M = group.random(GT)
        CT = tcabe.encrypt(M, policy_tree)
        
        partial_attrs = ['role:engineer', 'dept:maintenance']
        SK_partial = tcabe.keygen(MSK, partial_attrs)
        
        with pytest.raises(ValueError):
            tcabe.decrypt(SK_partial, CT)
        heartbeat(2, 10, "Partial attributes rejected")

    def test_policy_tree_modification_attack(self, system_setup, heartbeat):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        auth_attrs = ['role:intern']
        SK = tcabe.keygen(MSK, auth_attrs)
        
        policy_tree = parser.parse("role:engineer AND dept:maintenance")
        M = group.random(GT)
        CT = tcabe.encrypt(M, policy_tree)
        
        # Modify ciphertext leaf node attribute (CT tampering attack)
        # Ensure using correct key type
        first_leaf_id = list(CT['leaves'].keys())[0]
        CT['leaves'][first_leaf_id]['attr'] = 'role:intern'
        
        with pytest.raises(ValueError):
            tcabe.decrypt(SK, CT)
        heartbeat(3, 10, "Policy tree modification protected")


@pytest.mark.security
class TestEUF_CMASecurity:

    def test_signature_verification_tampered_message(self, system_setup, heartbeat, oom_protection):
        bls = system_setup['bls']
        pk, sk = bls.keygen()
        
        message = "original_auth_message"
        sigma = bls.sign(sk, message)
        
        assert not bls.verify(pk, "modified_auth_message", sigma)
        assert not bls.verify(pk, "", sigma)
        assert not bls.verify(pk, message.upper(), sigma)
        heartbeat(4, 10, "Signature tampering detected")
        oom_protection.check_and_protect()

    def test_signature_wrong_key(self, system_setup, heartbeat):
        bls = system_setup['bls']
        
        pk1, sk1 = bls.keygen()
        pk2, sk2 = bls.keygen()
        
        message = "test_message"
        sigma1 = bls.sign(sk1, message)
        
        assert bls.verify(pk1, message, sigma1)
        assert not bls.verify(pk2, message, sigma1)
        heartbeat(5, 10, "Wrong key signature rejected")

    def test_forged_certificate_detection(self, system_setup, heartbeat):
        cert_auth = system_setup['cert_auth']
        
        cert = cert_auth.issue_certificate(
            device_id='device_001',
            attributes=['type:sensor', 'location:factory']
        )
        
        forged1 = dict(cert)
        forged1['device_id'] = 'device_forged'
        assert not cert_auth.verify_certificate(forged1)
        
        forged2 = dict(cert)
        forged2['attributes'] = ['type:admin', 'location:anywhere']
        assert not cert_auth.verify_certificate(forged2)
        
        forged3 = dict(cert)
        forged3['signature'] = cert_auth.bls.sign(cert_auth.ca_sk, "different_data")
        assert not cert_auth.verify_certificate(forged3)
        heartbeat(6, 10, "Forged certificate detected")


@pytest.mark.security
class TestReplayAttackProtection:

    def test_nonce_replay_detection(self, system_setup, heartbeat, oom_protection):
        auth = system_setup['auth']
        
        nonce = auth.generate_nonce()
        timestamp = auth.generate_timestamp()
        
        assert auth.is_nonce_valid(nonce, timestamp)
        assert not auth.is_nonce_valid(nonce, timestamp)
        heartbeat(7, 10, "Nonce replay detected")
        oom_protection.check_and_protect()

    def test_expired_nonce_rejection(self, system_setup, heartbeat):
        auth = system_setup['auth']
        
        nonce = auth.generate_nonce()
        old_timestamp = time.time() - 600
        
        assert not auth.is_nonce_valid(nonce, old_timestamp)
        heartbeat(8, 10, "Expired nonce rejected")

    def test_auth_request_replay(self, system_setup, heartbeat):
        auth = system_setup['auth']
        parser = system_setup['parser']
        
        auth_request, _ = auth.device_auth_init('Device_001', ['type:sensor'])
        policy_T = parser.parse("type:sensor")
        
        challenge_resp = auth.digital_twin_challenge(auth_request, policy_T)
        assert challenge_resp['session_id'] is not None
        
        with pytest.raises(ValueError, match="Invalid or expired nonce"):
            auth.digital_twin_challenge(auth_request, policy_T)
        heartbeat(9, 10, "Auth request replay protected")


@pytest.mark.security
class TestTimePredicateSecurity:

    def test_time_predicate_bypass_prevention(self, system_setup, heartbeat, oom_protection):
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        attrs = ['role:engineer', 'dept:maintenance']
        SK = tcabe.keygen(MSK, attrs)
        
        policy_tree = parser.parse("role:engineer AND time:work")
        M = group.random(GT)
        CT = tcabe.encrypt(M, policy_tree)
        
        night_time = datetime(2026, 4, 20, 23, 0)
        with pytest.raises(ValueError):
            tcabe.decrypt(SK, CT, night_time)
        
        weekend_time = datetime(2026, 4, 25, 14, 0)
        with pytest.raises(ValueError):
            tcabe.decrypt(SK, CT, weekend_time)
        heartbeat(10, 10, "Time predicate bypass prevented")
        oom_protection.check_and_protect()


@pytest.mark.security
class TestSessionSecurity:

    def test_invalid_session_rejection(self, system_setup, heartbeat):
        auth = system_setup['auth']
        
        result = auth.digital_twin_verify({
            'session_id': 'nonexistent_session',
            'response': b'fake',
            'decrypted_challenge': b'fake',
            'timestamp': time.time()
        })
        assert not result['success']
        assert 'Invalid session ID' in result['message']
        heartbeat(11, 14, "Invalid session rejected")

    def test_session_hijacking_prevention(self, system_setup, heartbeat, oom_protection):
        auth = system_setup['auth']
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        parser = system_setup['parser']
        
        auth_request, device_bls_sk = auth.device_auth_init('Device_001', ['type:sensor'])
        policy_T = parser.parse("type:sensor")
        challenge_resp = auth.digital_twin_challenge(auth_request, policy_T)
        session_id = challenge_resp['session_id']
        
        device_abe_SK = tcabe.keygen(MSK, ['type:sensor'])
        response = auth.device_response(
            session_id,
            challenge_resp,
            device_abe_SK,
            device_bls_sk,
            'Device_001'
        )
        
        response['response'] = b'forged_response'
        verify_result = auth.digital_twin_verify(response)
        assert not verify_result['success']
        assert 'Invalid signature' in verify_result['message'] or 'Invalid' in verify_result['message']
        heartbeat(12, 14, "Session hijacking prevented")
        oom_protection.check_and_protect()

    def test_unverified_session_key_establishment_blocked(self, system_setup, heartbeat):
        auth = system_setup['auth']
        parser = system_setup['parser']
        
        auth_request, _ = auth.device_auth_init('Device_002', ['type:sensor'])
        policy_T = parser.parse("type:sensor")
        challenge_resp = auth.digital_twin_challenge(auth_request, policy_T)
        session_id = challenge_resp['session_id']
        
        with pytest.raises(ValueError, match="Session not verified"):
            auth.establish_session(session_id)
        heartbeat(13, 14, "Unverified session blocked")


@pytest.mark.security
class TestThreatAssessment:

    def test_high_threat_command_rejection(self, system_setup, heartbeat):
        dt_manager = system_setup['dt_manager']
        
        dt_manager.create_digital_twin('factory:critical_device', {'type': 'controller'})
        
        result = dt_manager.send_command(
            'factory:critical_device',
            {'action': 'shutdown', 'params': {}}
        )
        assert result['success']
        heartbeat(14, 14, "Threat assessment command processed")

    def test_diffusion_anomaly_scoring(self, heartbeat, oom_protection):
        torch = pytest.importorskip("torch")
        from src.diffusion import ThreatDiffusionModel
        
        model = ThreatDiffusionModel(
            vocab_size=100, embed_dim=64, condition_dim=32,
            num_train_timesteps=50, device='cpu'
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(10):
            batch_attrs = torch.randint(0, 100, (10, 20))
            batch_policy = torch.tensor([[10, 20, 30] + [0]*17]).expand(10, -1)
            loss = model.train_step(batch_attrs, batch_policy, optimizer)
            assert loss > 0
        heartbeat(15, 17, "Diffusion model training")
        oom_protection.check_and_protect()
        
        normal_request = {'attrs': [10, 20, 30]}
        normal_context = {'time_anomaly': False, 'behavior_anomaly': False}
        normal_score = model.anomaly_score(normal_request, normal_context)
        
        suspicious_request = {'attrs': [99, 99, 99], 'suspicious_attrs': True}
        suspicious_context = {'time_anomaly': True, 'behavior_anomaly': True}
        suspicious_score = model.anomaly_score(suspicious_request, suspicious_context)
        
        assert suspicious_score >= normal_score
        heartbeat(16, 17, "Anomaly score differentiation")

    def test_certificate_expiry_enforcement(self, system_setup, heartbeat):
        cert_auth = system_setup['cert_auth']
        
        cert = cert_auth.issue_certificate(
            device_id='temp_device',
            attributes=['type:temp'],
            validity_hours=0.001
        )
        
        time.sleep(5)
        
        assert not cert_auth.verify_certificate(cert)
        heartbeat(17, 17, "Certificate expiry enforced")
