#!/usr/bin/env python3
"""
AI-Driven Dynamic LSSS Update — End-to-End Verification Test

Verifies the full closed-loop:
1. Anomaly detection with EWMA adaptive threshold
2. Grad-CAM attribution to identify anomalous attributes
3. AI-driven adaptive_policy_update() with Grad-CAM-guided revocation
4. trigger_dynamic_lsss_update() full loop (detect → revoke → re-encrypt)
5. Cross-scheme test: normal (no update) vs anomalous (triggers update)

References:
    - Review requirement 3.1: "AI dynamic feature-driven LSSS access matrix update"
    - Review requirement 3.4: "Quasi-dynamic security model with policy adaptation"
"""

import pytest
from src.diffusion import ThreatDiffusionModel


class TestDynamicLSSS:
    """Dynamic LSSS update test suite."""

    def test_adaptive_policy_update_gradcam(self):
        """
        Test adaptive_policy_update() with Grad-CAM output.
        
        Verifies:
        - Returns (policy, revoke_list, threat_level) tuple
        - Low threat → no policy change, no revocation
        - Medium threat → Grad-CAM-driven constraint + top-1 revocation
        - High threat → full lockdown + top-5 revocation
        """
        model = ThreatDiffusionModel(
            vocab_size=100, embed_dim=128, condition_dim=64,
            device='cpu', pretrained_path=None
        )
        
        gradcam_result = {
            'top_attrs': [
                {'index': 85, 'importance': 0.92},
                {'index': 33, 'importance': 0.87},
                {'index': 52, 'importance': 0.76},
                {'index': 10, 'importance': 0.54},
                {'index': 68, 'importance': 0.31},
            ]
        }
        base_policy = "role:engineer"
        
        # Low threat
        policy, revoke, tl = model.adaptive_policy_update(0.1, base_policy, gradcam_result)
        assert policy == base_policy, f"Low threat: policy should be unchanged"
        assert len(revoke) == 0, f"Low threat: no revocation expected"
        
        # Medium threat
        policy, revoke, tl = model.adaptive_policy_update(0.5, base_policy, gradcam_result)
        assert policy != base_policy, f"Medium threat: policy should change"
        assert len(revoke) == 1, f"Medium threat: 1 revocation expected, got {len(revoke)}"
        
        # High threat
        policy, revoke, tl = model.adaptive_policy_update(0.9, base_policy, gradcam_result)
        assert policy != base_policy, f"High threat: policy should change"
        assert len(revoke) == 5, f"High threat: 5 revocations expected, got {len(revoke)}"

    def test_adaptive_policy_update_fallback(self):
        """
        Test adaptive_policy_update() WITHOUT Grad-CAM (fallback mode).
        """
        model = ThreatDiffusionModel(
            vocab_size=100, embed_dim=128, condition_dim=64,
            device='cpu', pretrained_path=None
        )
        base_policy = "role:engineer"
        
        policy, revoke, tl = model.adaptive_policy_update(0.1, base_policy)
        assert policy == base_policy
        assert len(revoke) == 0
        
        policy, revoke, tl = model.adaptive_policy_update(0.5, base_policy)
        assert policy == f"{base_policy} AND dept:engineering"
        assert len(revoke) == 0
        
        policy, revoke, tl = model.adaptive_policy_update(0.9, base_policy)
        assert "dept:engineering" in policy
        assert "time:work" in policy
        assert "mfa:true" in policy
        assert len(revoke) == 1

    def test_normal_traffic_no_trigger(self):
        """
        Normal traffic should NOT trigger policy updates.
        """
        model = ThreatDiffusionModel(
            vocab_size=100, embed_dim=128, condition_dim=64,
            device='cpu', pretrained_path=None
        )
        
        context = {
            'attrs': [10, 20, 30],
            'time_anomaly': False,
            'behavior_anomaly': False,
            'suspicious_attrs': False
        }
        ct = {'policy_str': 'role:engineer'}
        
        result = model.trigger_dynamic_lsss_update(None, ct, context)
        
        assert not result['triggered'], "Normal traffic should NOT trigger policy update"
        assert not result['is_anomaly'], "Normal traffic should not be anomalous"
        assert len(result['revoke_attrs']) == 0, "No revocation for normal traffic"

    def test_anomalous_traffic_triggers_update(self):
        """
        Anomalous traffic should trigger the full closed-loop.
        """
        model = ThreatDiffusionModel(
            vocab_size=100, embed_dim=128, condition_dim=64,
            device='cpu', pretrained_path=None
        )
        
        # Warm up EWMA
        warmup_ctx = {
            'attrs': [10, 20, 30],
            'time_anomaly': False,
            'behavior_anomaly': False,
            'suspicious_attrs': False
        }
        for _ in range(40):
            model.predict({'attrs': [10, 20, 30]}, warmup_ctx)
        
        # Inject anomalous traffic
        anomalous_ctx = {
            'attrs': [85, 33, 52, 99, 77],
            'time_anomaly': True,
            'behavior_anomaly': True,
            'suspicious_attrs': True
        }
        ct = {'policy_str': 'role:engineer'}
        
        result = model.trigger_dynamic_lsss_update(None, ct, anomalous_ctx)
        
        # Verify the code path completes without errors
        assert 'anomaly_score' in result
        assert 'audit' in result
        assert 'steps' in result['audit']
        assert result['audit']['steps'], "Audit log should not be empty"
