#!/usr/bin/env python3
"""
Diffusion Model Training Script for Scheme 4
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion import ThreatDiffusionModel

IOT_ATTR_VOCAB = {
    'type:sensor': 0, 'type:actuator': 1, 'type:controller': 2, 'type:gateway': 3,
    'protocol:tcp': 4, 'protocol:udp': 5, 'protocol:icmp': 6, 'protocol:http': 7,
    'port:80': 8, 'port:443': 9, 'port:8080': 10, 'port:22': 11,
    'access:read': 12, 'access:write': 13, 'access:admin': 14, 'access:execute': 15,
    'location:factory': 16, 'location:warehouse': 17, 'location:office': 18,
    'location:cloud': 19, 'network:internal': 20, 'network:dmz': 21,
    'network:external': 22, 'traffic:low': 23, 'traffic:medium': 24,
    'traffic:high': 25, 'state:active': 26, 'state:idle': 27, 'state:error': 28,
}

VOCAB_SIZE = len(IOT_ATTR_VOCAB)
MAX_ATTRS = 20

POLICIES = {
    'factory_access': {
        'required': ['type:sensor', 'location:factory', 'access:read'],
        'optional': ['protocol:tcp', 'network:internal', 'traffic:low'],
    },
    'admin_control': {
        'required': ['type:controller', 'access:admin', 'network:internal'],
        'optional': ['protocol:http', 'port:8080', 'state:active'],
    },
    'cloud_gateway': {
        'required': ['type:gateway', 'location:cloud', 'access:write'],
        'optional': ['protocol:https', 'port:443', 'traffic:high'],
    },
}


class IoTSyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, max_attrs=MAX_ATTRS, vocab_size=VOCAB_SIZE):
        self.num_samples = num_samples
        self.max_attrs = max_attrs
        self.vocab_size = vocab_size
        self.data, self.labels, self.policy_attrs = self._generate()

    def _generate(self):
        attr_list = list(IOT_ATTR_VOCAB.keys())
        policy_names = list(POLICIES.keys())
        data = []
        labels = []
        policy_attrs = []

        for i in range(self.num_samples):
            policy_name = policy_names[i % len(policy_names)]
            policy = POLICIES[policy_name]

            # Generate data attributes
            if i < self.num_samples * 0.7:
                attrs = list(policy['required'])
                remaining = self.max_attrs - len(attrs)
                if remaining > 0:
                    optional = policy['optional'][:remaining]
                    attrs.extend(optional)
                label = 1
            elif i < self.num_samples * 0.85:
                attrs = list(policy['required'])
                attrs = attrs[:-1] if len(attrs) > 1 else attrs
                fillers = [a for a in attr_list if a not in policy['required']]
                while len(attrs) < self.max_attrs:
                    attrs.append(np.random.choice(fillers))
                label = 0
            else:
                attrs = []
                while len(attrs) < self.max_attrs:
                    attrs.append(np.random.choice(attr_list))
                label = 0

            # Convert to indices
            indices = [IOT_ATTR_VOCAB.get(a, 0) for a in attrs[:self.max_attrs]]
            while len(indices) < self.max_attrs:
                indices.append(0)

            # Generate policy attributes (for conditional input)
            policy_attr = list(policy['required'])
            while len(policy_attr) < self.max_attrs:
                policy_attr.append('type:sensor')  # default padding
            policy_indices = [IOT_ATTR_VOCAB.get(a, 0) for a in policy_attr[:self.max_attrs]]

            data.append(indices)
            labels.append(label)
            policy_attrs.append(policy_indices)

        return (
            torch.tensor(data, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(policy_attrs, dtype=torch.long),
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.policy_attrs[idx]


def train_model(num_epochs=20, batch_size=32, lr=1e-4, device='cpu'):
    print("=" * 60)
    print("ThreatDiffusionModel Training")
    print("=" * 60)

    dataset = IoTSyntheticDataset(num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ThreatDiffusionModel(
        vocab_size=100,
        embed_dim=128,
        condition_dim=64,
        num_train_timesteps=100,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_attrs, batch_labels, batch_policy in dataloader:
            batch_attrs = batch_attrs.to(device)
            batch_policy = batch_policy.to(device)

            loss = model.train_step(batch_attrs, batch_policy, optimizer)
            epoch_loss += loss
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")

    print(f"\nTraining complete! Final loss: {losses[-1]:.6f}")
    return model, losses


def generate_adversarial_samples(model, device='cpu'):
    print("\n" + "-" * 60)
    print("Generating Adversarial Attribute Samples")
    print("-" * 60)

    policy_names = list(POLICIES.keys())
    results = {}

    for p_idx, policy_name in enumerate(policy_names):
        # Get required attributes of policy and convert to index tensor
        policy = POLICIES[policy_name]
        policy_attr = list(policy['required'])
        while len(policy_attr) < MAX_ATTRS:
            policy_attr.append('type:sensor')
        policy_indices = [IOT_ATTR_VOCAB.get(a, 0) for a in policy_attr[:MAX_ATTRS]]
        
        # Construct format expected by the model (1, max_attrs)
        policy_tensor = torch.tensor([policy_indices], device=device)
        samples = model.generate_adversarial(policy_tensor, n_samples=5, guidance_scale=1.5)

        if samples is not None:
            results[policy_name] = {
                'num_samples': 5,
                'sample_shape': list(samples.shape) if hasattr(samples, 'shape') else 'N/A',
                'target_policy': policy['required'],
            }
            print(f"  Policy '{policy_name}': Generated 5 adversarial samples")
        else:
            print(f"  Policy '{policy_name}': Generation returned None (model may need more training)")

    return results


def evaluate_anomaly_detection(model, device='cpu'):
    print("\n" + "-" * 60)
    print("Anomaly Detection Evaluation")
    print("-" * 60)

    test_dataset = IoTSyntheticDataset(num_samples=200)
    attr_list = list(IOT_ATTR_VOCAB.keys())

    legitimate_scores = []
    anomalous_scores = []

    for i in range(min(50, len(test_dataset))):
        attrs, label, _ = test_dataset[i]
        attr_indices = attrs.tolist()

        # Use format expected by the model {'attrs': [...]}
        auth_request = {'attrs': attr_indices}
        context = {
            'time_anomaly': False,
            'behavior_anomaly': False,
            'suspicious_attrs': False
        }

        score = model.anomaly_score(auth_request, context)

        if label == 1:
            legitimate_scores.append(score)
        else:
            anomalous_scores.append(score)

    if legitimate_scores:
        print(f"  Legitimate requests: avg_score={np.mean(legitimate_scores):.4f}, "
              f"std={np.std(legitimate_scores):.4f}")
    if anomalous_scores:
        print(f"  Anomalous requests:  avg_score={np.mean(anomalous_scores):.4f}, "
              f"std={np.std(anomalous_scores):.4f}")

    return {'legitimate': legitimate_scores, 'anomalous': anomalous_scores}


def save_results(losses, adversarial_results, anomaly_results, output_dir=None):
    if output_dir is None:
        output_dir = Path(__file__).parent / 'results' / 'diffusion_training'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_data = {
        'timestamp': datetime.now().isoformat(),
        'num_epochs': len(losses),
        'final_loss': losses[-1] if losses else None,
        'loss_history': losses,
        'adversarial_generation': adversarial_results,
        'anomaly_detection': {
            'legitimate_avg': float(np.mean(anomaly_results['legitimate'])) if anomaly_results.get('legitimate') else None,
            'anomalous_avg': float(np.mean(anomaly_results['anomalous'])) if anomaly_results.get('anomalous') else None,
        },
    }

    output_file = output_dir / 'training_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Vocab size: {VOCAB_SIZE}, Max attrs: {MAX_ATTRS}")

    model, losses = train_model(num_epochs=20, batch_size=32, lr=1e-4, device=device)

    adversarial_results = generate_adversarial_samples(model, device=device)

    anomaly_results = evaluate_anomaly_detection(model, device=device)

    save_results(losses, adversarial_results, anomaly_results)
    
    # Save trained model weights
    weights_dir = Path(__file__).parent / 'weights'
    weights_dir.mkdir(exist_ok=True)
    weights_path = weights_dir / 'threat_diffusion.pth'
    model.save_weights(str(weights_path))
    print(f"\nModel weights saved to: {weights_path}")

    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
