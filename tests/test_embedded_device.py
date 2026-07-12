"""
Scheme 4: Digital Twin Network Security Framework - Embedded Device Tests

Test performance on resource-constrained devices (e.g., Raspberry Pi, Arduino):
1. Device resource detection
2. Performance tests (Key Generation, Encrypt, Decrypt)
3. Memory usage tests
4. Comparison with desktop environment
5. Battery consumption estimation (mobile devices)

Test content:

- Can it run on small devices like Raspberry Pi?
- How does it perform in resource-constrained environments?

"""

import pytest
import time
import gc
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Any
from charm.toolbox.pairinggroup import GT

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class DeviceInfo:
    """Device Information Class"""
    
    @staticmethod
    def get_device_info():
        """Get device information"""
        import platform
        import psutil
        
        try:
            return {
                'device_type': DeviceInfo._detect_device_type(),
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': os.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
                'disk_available_gb': psutil.disk_usage('/').free / (1024**3)
            }
        except Exception as e:
            return {
                'device_type': 'unknown',
                'error': str(e)
            }
    
    @staticmethod
    def _detect_device_type():
        """Detect device type"""
        import platform
        
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Raspberry Pi detection
        if 'raspberry' in machine or 'arm' in machine:
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                if 'Raspberry Pi' in content:
                    return 'raspberry_pi'
            except:
                pass
        
        # Other device detection
        if system == 'linux':
            return 'linux'
        elif system == 'darwin':
            return 'macos'
        elif system == 'windows':
            return 'windows'
        else:
            return 'unknown'
    
    @staticmethod
    def is_embedded_device():
        """Determine if it is an embedded device"""
        info = DeviceInfo.get_device_info()
        device_type = info.get('device_type', 'unknown')
        memory = info.get('memory_total_gb', 16)
        
        # Devices with less than 4GB memory are considered embedded devices
        if memory < 4:
            return True
        
        # Specific device types
        if device_type in ['raspberry_pi', 'arm']:
            return True
        
        return False


@pytest.mark.embedded
@pytest.mark.slow
class TestEmbeddedDevice:
    """Embedded Device Tests"""
    
    ATTR_COUNTS = [10, 50, 100]  # Attribute counts on embedded devices
    
    @staticmethod
    def _save_embedded_data(data, filename):
        """Save embedded device test data"""
        output_dir = Path(__file__).parent.parent / 'experiments' / 'results' / 'embedded'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def test_device_info(self, heartbeat):
        """Test device information"""
        info = DeviceInfo.get_device_info()
        is_embedded = DeviceInfo.is_embedded_device()
        
        print("\n" + "=" * 80)
        print("Device Information")
        print("=" * 80)
        for key, value in info.items():
            print(f"{key:20}: {value}")
        print(f"is_embedded: {is_embedded}")
        
        self._save_embedded_data(info, 'device_info.json')
        heartbeat(1, 1, "Device information detection completed")
    
    def test_embedded_performance(self, system_setup, heartbeat, oom_protection):
        """Embedded Device Performance Test"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        results = {}
        total_steps = len(self.ATTR_COUNTS) * 3  # keygen, encrypt, decrypt
        step = 0
        
        for num_attrs in self.ATTR_COUNTS:
            gc.collect()
            
            # Generate attributes
            attrs = [f'attr_{i}' for i in range(num_attrs)]
            
            # Key Generation
            start = time.time()
            SK = tcabe.keygen(MSK, attrs)
            keygen_time = time.time() - start
            assert SK is not None
            
            step += 1
            heartbeat(step, total_steps, f'KeyGen: {num_attrs} attrs → {keygen_time:.3f}s')
            
            # Encrypt
            policy_str = ' AND '.join(attrs[:min(5, num_attrs)])  # Simplified policy
            policy_tree = parser.parse(policy_str)
            M = group.random(GT)
            
            start = time.time()
            CT = tcabe.encrypt(M, policy_tree)
            encrypt_time = time.time() - start
            assert CT is not None
            
            step += 1
            heartbeat(step, total_steps, f'Encrypt: {num_attrs} attrs → {encrypt_time:.3f}s')
            
            # Decrypt
            start = time.time()
            decrypted = tcabe.decrypt(SK, CT)
            decrypt_time = time.time() - start
            assert decrypted == M
            
            step += 1
            heartbeat(step, total_steps, f'Decrypt: {num_attrs} attrs → {decrypt_time:.3f}s')
            
            # Memory usage
            import psutil
            memory = psutil.virtual_memory()
            memory_used = memory.used / (1024**3)
            memory_percent = memory.percent
            
            results[num_attrs] = {
                'keygen_time': keygen_time,
                'encrypt_time': encrypt_time,
                'decrypt_time': decrypt_time,
                'memory_used_gb': memory_used,
                'memory_percent': memory_percent
            }
            
            oom_protection.check_and_protect()
        
        self._save_embedded_data(results, 'embedded_performance.json')
        
        print("\n" + "=" * 80)
        print("Embedded Device Performance Test Results (used for SCI paper Table 4)")
        print("=" * 80)
        print(f"{'AttributeCount':12} {'KeyGen(s)':12} {'Encrypt(s)':12} {'Decrypt(s)':12} {'MemoryUsed(GB)':15} {'MemoryUsage(%)':12}")
        print("-" * 80)
        for num_attrs, data in results.items():
            print(f"{num_attrs:12} {data['keygen_time']:11.3f} {data['encrypt_time']:11.3f} " +
                  f"{data['decrypt_time']:11.3f} {data['memory_used_gb']:14.2f} {data['memory_percent']:11.1f}")
    
    def test_embedded_vs_desktop(self, system_setup, heartbeat, oom_protection):
        """Current device performance benchmark (actual measurement)"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        device_info = DeviceInfo.get_device_info()

        results = {}
        total_steps = len(self.ATTR_COUNTS)
        step = 0

        for num_attrs in self.ATTR_COUNTS:
            gc.collect()
            attrs = [f'attr_{i}' for i in range(num_attrs)]

            start = time.time()
            SK = tcabe.keygen(MSK, attrs)
            keygen_time = time.time() - start

            policy_str = ' AND '.join(attrs[:min(5, num_attrs)])
            policy_tree = parser.parse(policy_str)
            M = group.random(GT)

            start = time.time()
            CT = tcabe.encrypt(M, policy_tree)
            encrypt_time = time.time() - start

            start = time.time()
            decrypted = tcabe.decrypt(SK, CT)
            decrypt_time = time.time() - start
            assert decrypted == M

            results[num_attrs] = {
                'keygen_time': round(keygen_time, 4),
                'encrypt_time': round(encrypt_time, 4),
                'decrypt_time': round(decrypt_time, 4)
            }

            step += 1
            heartbeat(step, total_steps, f'{num_attrs} attrs: KG={keygen_time:.3f}s, Enc={encrypt_time:.3f}s, Dec={decrypt_time:.3f}s')
            oom_protection.check_and_protect()

        self._save_embedded_data({
            'device_info': device_info,
            'performance': results
        }, 'device_performance_benchmark.json')

        print("\n" + "=" * 80)
        print(f"Current Device Performance Benchmark: {device_info.get('platform', 'unknown')}")
        print("Note: Cross-device comparison requires running this test on target devices separately")
        print("=" * 80)
        print(f"{'AttributeCount':12} {'KeyGen(s)':12} {'Encrypt(s)':12} {'Decrypt(s)':12}")
        print("-" * 80)
        for num_attrs, data in results.items():
            print(f"{num_attrs:12} {data['keygen_time']:11.4f} {data['encrypt_time']:11.4f} {data['decrypt_time']:11.4f}")
        print("=" * 80)
    
    def test_battery_estimation(self, system_setup, heartbeat, oom_protection):
        """Battery consumption estimation - theoretical estimation based on actual operation time"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']

        results = {}
        total_steps = len(self.ATTR_COUNTS)
        step = 0

        for num_attrs in self.ATTR_COUNTS:
            gc.collect()
            attrs = [f'attr_{i}' for i in range(num_attrs)]
            policy_str = ' AND '.join(attrs[:min(5, num_attrs)])
            policy_tree = parser.parse(policy_str)
            M = group.random(GT)

            start = time.time()
            SK = tcabe.keygen(MSK, attrs)
            keygen_time = time.time() - start

            start = time.time()
            CT = tcabe.encrypt(M, policy_tree)
            encrypt_time = time.time() - start

            start = time.time()
            decrypted = tcabe.decrypt(SK, CT)
            decrypt_time = time.time() - start
            assert decrypted == M

            ops_per_hour_keygen = 3600 / keygen_time if keygen_time > 0 else 0
            ops_per_hour_encrypt = 3600 / encrypt_time if encrypt_time > 0 else 0
            ops_per_hour_decrypt = 3600 / decrypt_time if decrypt_time > 0 else 0

            results[num_attrs] = {
                'keygen_time_s': round(keygen_time, 4),
                'encrypt_time_s': round(encrypt_time, 4),
                'decrypt_time_s': round(decrypt_time, 4),
                'ops_per_hour_keygen': round(ops_per_hour_keygen, 0),
                'ops_per_hour_encrypt': round(ops_per_hour_encrypt, 0),
                'ops_per_hour_decrypt': round(ops_per_hour_decrypt, 0)
            }

            step += 1
            heartbeat(step, total_steps, f'{num_attrs} attrs: KG={keygen_time:.3f}s, Enc={encrypt_time:.3f}s, Dec={decrypt_time:.3f}s')
            oom_protection.check_and_protect()

        self._save_embedded_data(results, 'battery_estimation.json')

        print("\n" + "=" * 80)
        print("Operation Throughput Estimation (based on actual measurement)")
        print("Note: Actual battery life depends on hardware power consumption, needs measurement on target device")
        print("=" * 80)
        print(f"{'AttributeCount':12} {'KeyGen(op/h)':15} {'Encrypt(op/h)':15} {'Decrypt(op/h)':15}")
        print("-" * 80)
        for num_attrs, data in results.items():
            print(f"{num_attrs:12} {data['ops_per_hour_keygen']:14.0f} {data['ops_per_hour_encrypt']:14.0f} {data['ops_per_hour_decrypt']:14.0f}")
        print("=" * 80)
    
    def test_generate_embedded_table(self, system_setup, heartbeat, oom_protection):
        """Generate Embedded Device Test Table (used for SCI paper Table 4) - using actual measurement data"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        device_info = DeviceInfo.get_device_info()

        results = {}
        total_steps = len(self.ATTR_COUNTS)
        step = 0

        for num_attrs in self.ATTR_COUNTS:
            gc.collect()
            attrs = [f'attr_{i}' for i in range(num_attrs)]

            start = time.time()
            SK = tcabe.keygen(MSK, attrs)
            keygen_time = time.time() - start

            policy_str = ' AND '.join(attrs[:min(5, num_attrs)])
            policy_tree = parser.parse(policy_str)
            M = group.random(GT)

            start = time.time()
            CT = tcabe.encrypt(M, policy_tree)
            encrypt_time = time.time() - start

            start = time.time()
            decrypted = tcabe.decrypt(SK, CT)
            decrypt_time = time.time() - start
            assert decrypted == M

            import psutil
            memory = psutil.virtual_memory()

            results[num_attrs] = {
                'keygen_time': round(keygen_time, 4),
                'encrypt_time': round(encrypt_time, 4),
                'decrypt_time': round(decrypt_time, 4),
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_percent': round(memory.percent, 1)
            }

            step += 1
            heartbeat(step, total_steps, f'{num_attrs} attrs: KG={keygen_time:.3f}s, Enc={encrypt_time:.3f}s, Dec={decrypt_time:.3f}s')
            oom_protection.check_and_protect()

        self._save_embedded_data({
            'device_info': device_info,
            'results': results
        }, 'embedded_device_table.json')

        device_name = device_info.get('platform', 'unknown')
        print("\n" + "=" * 100)
        print(f"Table 4: Embedded Device Test Results (used for SCI paper) - Current Device: {device_name}")
        print("Note: Data for other devices (Raspberry Pi, Arduino, etc.) needs to run this test on corresponding devices")
        print("=" * 100)
        print(f"{'DeviceType':20} {'AttributeCount':12} {'KeyGen(s)':12} {'Encrypt(s)':12} {'Decrypt(s)':12} {'MemoryUsed(GB)':15}")
        print("-" * 100)
        for num_attrs, data in results.items():
            print(f"{device_name:20} {num_attrs:12} {data['keygen_time']:11.4f} {data['encrypt_time']:11.4f} {data['decrypt_time']:11.4f} {data['memory_used_gb']:14.2f}")
        print("=" * 100)
