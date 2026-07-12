#!/usr/bin/env python3
"""
Grad-CAM 可解释性溯源模块验证脚本

验证：
1. generate_gradcam 能否正常执行（hook注册/移除正确）
2. 输出结构是否完整（heatmap_embed, heatmap_attr, anomaly_score, top_attrs）
3. 模型的随机初始化状态不影响Grad-CAM的基本流程

用法：
    python test_gradcam.py
"""

import sys
import os
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent / 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from diffusion import ThreatDiffusionModel


def main():
    print("=" * 60)
    print("Grad-CAM Explainability Module — Verification Test")
    print("=" * 60)
    
    # ---- Step 1: 创建模型 (随机初始化) ----
    print("\n[Step 1] Creating model (random init, no pretrained weights)...")
    model = ThreatDiffusionModel(
        vocab_size=100,
        embed_dim=128,
        condition_dim=64,
        device='cpu',
        pretrained_path=None
    )
    print(f"  Model created: embed_dim={model.embed_dim}, vocab_size={model.vocab_size}")
    print(f"  Model trained status: {model.is_trained}")
    
    # ---- Step 2: 构造模拟认证请求 ----
    print("\n[Step 2] Creating synthetic auth request...")
    auth_request = {
        'attrs': [10, 25, 33, 47, 52, 68, 71, 85, 92, 3, 15, 28, 41, 56, 63, 77, 84, 96, 7, 19],
        'timestamp': '2026-01-15T10:30:00',
        'device_id': 'nano_pi_r4s_01',
    }
    print(f"  Attributes: {auth_request['attrs']}")
    
    results_dir = Path(__file__).resolve().parent / 'results' / 'ablation'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- Step 3: 生成Grad-CAM热力图 ----
    print("\n[Step 3] Generating Grad-CAM heatmap...")
    try:
        result = model.generate_gradcam(auth_request)
        print("  ✓ generate_gradcam() completed successfully")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ---- Step 4: 检查输出完整性 ----
    print("\n[Step 4] Validating output structure...")
    
    expected_keys = ['heatmap_embed', 'heatmap_attr', 'anomaly_score', 'top_attrs']
    for key in expected_keys:
        if key in result:
            print(f"  ✓ '{key}' present")
        else:
            print(f"  ✗ '{key}' MISSING!")
    
    # Validate shapes
    heatmap_embed = result['heatmap_embed']
    heatmap_attr = result['heatmap_attr']
    top_attrs = result['top_attrs']
    print(f"\n  heatmap_embed: len={len(heatmap_embed)} (expected 128)")
    print(f"  heatmap_attr:  len={len(heatmap_attr)} (expected 100)")
    print(f"  top_attrs:     count={len(top_attrs)} (expected 5)")
    
    assert len(result['heatmap_embed']) == 128, "heatmap_embed should be 128-dim"
    assert len(result['heatmap_attr']) == 100, "heatmap_attr should be 100-dim (vocab_size)"
    assert len(result['top_attrs']) == 5, "top_attrs should have 5 entries"
    print("  ✓ All shapes match expected")
    
    # ---- Step 5: 显示热力图统计数据 ----
    print(f"\n[Step 5] Heatmap statistics:")
    print(f"  Anomaly Score: {result['anomaly_score']:.6f}")
    print(f"  Heatmap (embed) range: [{min(heatmap_embed):.4f}, {max(heatmap_embed):.4f}]")
    print(f"  Heatmap (attr)  range: [{min(heatmap_attr):.4f}, {max(heatmap_attr):.4f}]")
    positive = sum(1 for v in heatmap_embed if v > 0.1)
    print(f"  High-importance dims (>0.1): {positive}/128")
    
    # ---- Step 6: 显示Top-5属性 ----
    print(f"\n[Step 6] Top-5 Most Influential Attributes:")
    for i, attr in enumerate(top_attrs):
        print(f"  {i+1}. Attribute[{attr['index']}] — importance: {attr['importance']:.4f}")
    
    # ---- Step 7: 验证多次调用稳定性 ----
    print(f"\n[Step 7] Stability test (5 consecutive calls)...")
    scores = []
    for i in range(5):
        r = model.generate_gradcam(auth_request)
        scores.append(r['anomaly_score'])
    score_std = __import__('statistics').stdev(scores)
    print(f"  Anomaly scores: {[f'{s:.6f}' for s in scores]}")
    print(f"  Std dev: {score_std:.6f}")
    print(f"  {'✓ Stable' if score_std < 1e-4 else '⚠ May have randomness'}")
    
    # ---- Step 8: 格式化报告 ----
    print(f"\n[Step 8] Formatted report:")
    print("-" * 40)
    report = model.format_gradcam_report(result)
    print(report)
    
    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("✓ All Grad-CAM tests passed!")
    print(f"{'=' * 60}")
    
    # ---- 保存结果 ----
    results_path = results_dir / 'gradcam_result.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {results_path}")


if __name__ == '__main__':
    import json
    main()
