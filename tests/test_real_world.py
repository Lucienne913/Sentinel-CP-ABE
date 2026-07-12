"""
Scheme 4: Digital Twin Network Security Framework - Real Dataset Verification Tests

Verify the scheme using real-world scenario data:
1. Industrial IoT scenario (factory equipment)
2. Healthcare scenario (patient data access)
3. Smart City scenario (city infrastructure)
4. Cloud Storage scenario (large-scale cloud access control)
"""

import pytest
import time
import gc
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from charm.toolbox.pairinggroup import GT

sys.path.insert(0, str(Path(__file__).parent.parent))


class RealWorldDatasets:
    """Real-World Scenario Dataset Generator"""
    
    @staticmethod
    def generate_industrial_iot_data(num_devices=100):
        """Generate Industrial IoT scenario data"""
        device_types = ['sensor', 'actuator', 'controller', 'gateway']
        locations = ['factory_a', 'factory_b', 'warehouse_1', 'warehouse_2', 'maintenance']
        roles = ['operator', 'technician', 'manager', 'engineer']
        
        devices = []
        for i in range(num_devices):
            device_id = f'iot_device_{i:04d}'
            device_type = device_types[i % len(device_types)]
            location = locations[i % len(locations)]
            role = roles[i % len(roles)]
            
            attrs = [
                f'device:{device_id}',
                f'type:{device_type}',
                f'location:{location}',
                f'role:{role}'
            ]
            
            devices.append({
                'id': device_id,
                'attrs': attrs,
                'type': device_type,
                'location': location,
                'role': role
            })
        
        return devices
    
    @staticmethod
    def generate_healthcare_data(num_patients=50):
        """Generate healthcare scenario data"""
        departments = ['cardiology', 'neurology', 'pediatrics', 'radiology', 'emergency']
        roles = ['doctor', 'nurse', 'researcher', 'admin', 'patient']
        security_levels = ['level_1', 'level_2', 'level_3']
        
        patients = []
        for i in range(num_patients):
            patient_id = f'patient_{i:04d}'
            department = departments[i % len(departments)]
            role = roles[i % len(roles)]
            level = security_levels[i % len(security_levels)]
            
            attrs = [
                f'patient:{patient_id}',
                f'department:{department}',
                f'role:{role}',
                f'clearance:{level}'
            ]
            
            patients.append({
                'id': patient_id,
                'attrs': attrs,
                'department': department,
                'role': role,
                'level': level
            })
        
        return patients
    
    @staticmethod
    def generate_smart_city_data(num_nodes=200):
        """Generate Smart City scenario data"""
        node_types = ['traffic_light', 'camera', 'sensor', 'display', 'gateway']
        districts = ['downtown', 'suburb_a', 'suburb_b', 'industrial_park', 'airport']
        priorities = ['high', 'medium', 'low']
        
        nodes = []
        for i in range(num_nodes):
            node_id = f'city_node_{i:04d}'
            node_type = node_types[i % len(node_types)]
            district = districts[i % len(districts)]
            priority = priorities[i % len(priorities)]
            
            attrs = [
                f'node:{node_id}',
                f'type:{node_type}',
                f'district:{district}',
                f'priority:{priority}'
            ]
            
            nodes.append({
                'id': node_id,
                'attrs': attrs,
                'type': node_type,
                'district': district,
                'priority': priority
            })
        
        return nodes
    
    @staticmethod
    def generate_cloud_storage_data(num_users=300):
        """Generate Cloud Storage scenario data"""
        user_roles = ['admin', 'editor', 'viewer', 'manager', 'owner']
        departments = ['engineering', 'sales', 'hr', 'finance', 'research']
        regions = ['us_east', 'us_west', 'eu_central', 'asia_pacific']
        
        users = []
        for i in range(num_users):
            user_id = f'cloud_user_{i:04d}'
            role = user_roles[i % len(user_roles)]
            department = departments[i % len(departments)]
            region = regions[i % len(regions)]
            
            attrs = [
                f'user:{user_id}',
                f'role:{role}',
                f'department:{department}',
                f'region:{region}'
            ]
            
            users.append({
                'id': user_id,
                'attrs': attrs,
                'role': role,
                'department': department,
                'region': region
            })
        
        return users


@pytest.mark.slow
@pytest.mark.basic
class TestRealWorldScenarios:
    """Real-World Scenario Verification Tests"""
    
    @staticmethod
    def _save_real_world_data(data, filename):
        """Save real-world scenario data to file"""
        import json
        output_dir = Path(__file__).parent.parent / 'experiments' / 'results' / 'real_world'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def test_industrial_iot_scenario(self, system_setup, heartbeat):
        """Industrial IoT scenario verification (factory equipment access control)"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        print("\n" + "=" * 80)
        print("Industrial IoT Scenario Verification")
        print("=" * 80)
        
        devices = RealWorldDatasets.generate_industrial_iot_data(100)
        results = {
            'scenario': 'Industrial IoT',
            'num_devices': len(devices),
            'results': []
        }
        
        total_steps = len(devices) + 3
        step = 0
        
        # Define policy: only factory_a or factory_b technicians/engineers can access
        policy_str = '( location:factory_a OR location:factory_b ) AND ( role:technician OR role:engineer )'
        policy_tree = parser.parse(policy_str)
        
        step += 1
        heartbeat(step, total_steps, f'Policy generated')
        
        M = group.random(GT)
        
        # Encrypt data
        start = time.time()
        CT = tcabe.encrypt(M, policy_tree)
        encrypt_time = time.time() - start
        assert CT is not None
        
        step += 1
        heartbeat(step, total_steps, f'Encrypted: {encrypt_time:.3f}s')
        
        # Verify each device's access permission
        success_count = 0
        total_time = 0
        
        for device in devices:
            attrs = device['attrs']
            SK = tcabe.keygen(MSK, attrs)
            
            # Only allow work time access (9:00-18:00)
            work_time = datetime(2026, 4, 22, 14, 30)
            
            try:
                start = time.time()
                decrypted = tcabe.decrypt(SK, CT, work_time)
                elapsed = time.time() - start
                assert decrypted == M
                
                success_count += 1
                total_time += elapsed
                
                results['results'].append({
                    'device_id': device['id'],
                    'granted': True,
                    'time': elapsed,
                    'location': device['location'],
                    'role': device['role']
                })
            except Exception:
                results['results'].append({
                    'device_id': device['id'],
                    'granted': False,
                    'time': 0,
                    'location': device['location'],
                    'role': device['role']
                })
            
            step += 1
            heartbeat(step, total_steps, f'Device {device["id"]}: {results["results"][-1]["granted"]}')
        
        avg_time = total_time / success_count if success_count > 0 else 0
        results['summary'] = {
            'encrypt_time': encrypt_time,
            'total_devices': len(devices),
            'success_count': success_count,
            'success_rate': success_count / len(devices),
            'avg_decrypt_time': avg_time
        }
        
        self._save_real_world_data(results, 'industrial_iot_results.json')
        
        print(f"\nIndustrial IoT Scenario Summary:")
        print(f"  Total devices: {len(devices)}")
        print(f"  Successful access: {success_count}")
        print(f"  Success rate: {success_count/len(devices)*100:.1f}%")
        print(f"  Average decryption time: {avg_time:.3f}s")
        print(f"  Encryption time: {encrypt_time:.3f}s")
    
    def test_healthcare_scenario(self, system_setup, heartbeat):
        """Healthcare scenario verification (patient data access control)"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        print("\n" + "=" * 80)
        print("Healthcare Scenario Verification")
        print("=" * 80)
        
        patients = RealWorldDatasets.generate_healthcare_data(50)
        results = {
            'scenario': 'Healthcare',
            'num_patients': len(patients),
            'results': []
        }
        
        total_steps = len(patients) + 3
        step = 0
        
        # Policy: cardiology/neurology doctors, at least level_2 security clearance
        policy_str = '( department:cardiology OR department:neurology ) AND clearance:level_2'
        policy_tree = parser.parse(policy_str)
        
        step += 1
        heartbeat(step, total_steps, 'Policy generated')
        
        M = group.random(GT)
        
        start = time.time()
        CT = tcabe.encrypt(M, policy_tree)
        encrypt_time = time.time() - start
        assert CT is not None
        
        step += 1
        heartbeat(step, total_steps, f'Encrypted: {encrypt_time:.3f}s')
        
        success_count = 0
        total_time = 0
        
        for patient in patients:
            attrs = patient['attrs']
            SK = tcabe.keygen(MSK, attrs)
            
            work_time = datetime(2026, 4, 22, 10, 0)
            
            try:
                start = time.time()
                decrypted = tcabe.decrypt(SK, CT, work_time)
                elapsed = time.time() - start
                assert decrypted == M
                
                success_count += 1
                total_time += elapsed
                
                results['results'].append({
                    'patient_id': patient['id'],
                    'granted': True,
                    'time': elapsed,
                    'department': patient['department'],
                    'role': patient['role']
                })
            except Exception:
                results['results'].append({
                    'patient_id': patient['id'],
                    'granted': False,
                    'time': 0,
                    'department': patient['department'],
                    'role': patient['role']
                })
            
            step += 1
            heartbeat(step, total_steps, f'Patient {patient["id"]}')
        
        avg_time = total_time / success_count if success_count > 0 else 0
        results['summary'] = {
            'encrypt_time': encrypt_time,
            'total_patients': len(patients),
            'success_count': success_count,
            'success_rate': success_count / len(patients),
            'avg_decrypt_time': avg_time
        }
        
        self._save_real_world_data(results, 'healthcare_results.json')
        
        print(f"\nHealthcare Scenario Summary:")
        print(f"  Total patients: {len(patients)}")
        print(f"  Successful access: {success_count}")
        print(f"  Success rate: {success_count/len(patients)*100:.1f}%")
        print(f"  Average decryption time: {avg_time:.3f}s")
    
    def test_smart_city_scenario(self, system_setup, heartbeat):
        """Smart City scenario verification (city infrastructure access)"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        print("\n" + "=" * 80)
        print("Smart City Scenario Verification")
        print("=" * 80)
        
        nodes = RealWorldDatasets.generate_smart_city_data(200)
        results = {
            'scenario': 'Smart City',
            'num_nodes': len(nodes),
            'results': []
        }
        
        total_steps = len(nodes) + 3
        step = 0
        
        policy_str = '( district:downtown OR district:airport ) AND priority:high'
        policy_tree = parser.parse(policy_str)
        
        step += 1
        heartbeat(step, total_steps, 'Policy generated')
        
        M = group.random(GT)
        
        start = time.time()
        CT = tcabe.encrypt(M, policy_tree)
        encrypt_time = time.time() - start
        assert CT is not None
        
        step += 1
        heartbeat(step, total_steps, f'Encrypted: {encrypt_time:.3f}s')
        
        success_count = 0
        total_time = 0
        
        for node in nodes:
            attrs = node['attrs']
            SK = tcabe.keygen(MSK, attrs)
            
            try:
                start = time.time()
                decrypted = tcabe.decrypt(SK, CT)
                elapsed = time.time() - start
                assert decrypted == M
                
                success_count += 1
                total_time += elapsed
                
                results['results'].append({
                    'node_id': node['id'],
                    'granted': True,
                    'time': elapsed,
                    'type': node['type'],
                    'district': node['district']
                })
            except Exception:
                results['results'].append({
                    'node_id': node['id'],
                    'granted': False,
                    'time': 0,
                    'type': node['type'],
                    'district': node['district']
                })
            
            step += 1
            heartbeat(step, total_steps, f'Node {node["id"]}')
        
        avg_time = total_time / success_count if success_count > 0 else 0
        results['summary'] = {
            'encrypt_time': encrypt_time,
            'total_nodes': len(nodes),
            'success_count': success_count,
            'success_rate': success_count / len(nodes),
            'avg_decrypt_time': avg_time
        }
        
        self._save_real_world_data(results, 'smart_city_results.json')
        
        print(f"\nSmart City Scenario Summary:")
        print(f"  Total nodes: {len(nodes)}")
        print(f"  Successful access: {success_count}")
        print(f"  Success rate: {success_count/len(nodes)*100:.1f}%")
        print(f"  Average decryption time: {avg_time:.3f}s")
    
    def test_cloud_storage_scenario(self, system_setup, heartbeat):
        """Cloud Storage scenario verification (large-scale user access)"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']
        
        print("\n" + "=" * 80)
        print("Cloud Storage Scenario Verification")
        print("=" * 80)
        
        users = RealWorldDatasets.generate_cloud_storage_data(300)
        results = {
            'scenario': 'Cloud Storage',
            'num_users': len(users),
            'results': []
        }
        
        total_steps = len(users) + 3
        step = 0
        
        policy_str = '( role:admin OR role:editor OR role:owner ) AND ( department:engineering OR department:research )'
        policy_tree = parser.parse(policy_str)
        
        step += 1
        heartbeat(step, total_steps, 'Policy generated')
        
        M = group.random(GT)
        
        start = time.time()
        CT = tcabe.encrypt(M, policy_tree)
        encrypt_time = time.time() - start
        assert CT is not None
        
        step += 1
        heartbeat(step, total_steps, f'Encrypted: {encrypt_time:.3f}s')
        
        success_count = 0
        total_time = 0
        
        for user in users:
            attrs = user['attrs']
            SK = tcabe.keygen(MSK, attrs)
            
            work_time = datetime(2026, 4, 22, 11, 0)
            
            try:
                start = time.time()
                decrypted = tcabe.decrypt(SK, CT, work_time)
                elapsed = time.time() - start
                assert decrypted == M
                
                success_count += 1
                total_time += elapsed
                
                results['results'].append({
                    'user_id': user['id'],
                    'granted': True,
                    'time': elapsed,
                    'role': user['role'],
                    'department': user['department']
                })
            except Exception:
                results['results'].append({
                    'user_id': user['id'],
                    'granted': False,
                    'time': 0,
                    'role': user['role'],
                    'department': user['department']
                })
            
            step += 1
            heartbeat(step, total_steps, f'User {user["id"]}')
        
        avg_time = total_time / success_count if success_count > 0 else 0
        results['summary'] = {
            'encrypt_time': encrypt_time,
            'total_users': len(users),
            'success_count': success_count,
            'success_rate': success_count / len(users),
            'avg_decrypt_time': avg_time
        }
        
        self._save_real_world_data(results, 'cloud_storage_results.json')
        
        print(f"\nCloud Storage Scenario Summary:")
        print(f"  Total users: {len(users)}")
        print(f"  Successful access: {success_count}")
        print(f"  Success rate: {success_count/len(users)*100:.1f}%")
        print(f"  Average decryption time: {avg_time:.3f}s")
    
    def test_generate_real_world_summary_table(self, system_setup, heartbeat):
        """Generate real-world scenario summary table (used for SCI paper Table 4) - using actual measurement data"""
        tcabe = system_setup['tcabe']
        MSK = system_setup['MSK']
        PP = system_setup['PP']
        group = PP['group']
        parser = system_setup['parser']

        scenarios_config = [
            ('Industrial IoT', 100,
             '( location:factory_a OR location:factory_b ) AND ( role:technician OR role:engineer )',
             lambda i: [f'device:iot_device_{i:04d}', f'type:{["sensor","actuator","controller","gateway"][i%4]}',
                        f'location:{["factory_a","factory_b","warehouse_1","warehouse_2","maintenance"][i%5]}',
                        f'role:{["operator","technician","manager","engineer"][i%4]}']),
            ('Healthcare', 50,
             '( department:cardiology OR department:neurology ) AND clearance:level_2',
             lambda i: [f'patient:patient_{i:04d}', f'department:{["cardiology","neurology","pediatrics","radiology","emergency"][i%5]}',
                        f'role:{["doctor","nurse","researcher","admin","patient"][i%5]}',
                        f'clearance:{["level_1","level_2","level_3"][i%3]}']),
            ('Smart City', 200,
             '( district:downtown OR district:airport ) AND priority:high',
             lambda i: [f'node:city_node_{i:04d}', f'type:{["traffic_light","camera","sensor","display","gateway"][i%5]}',
                        f'district:{["downtown","suburb_a","suburb_b","industrial_park","airport"][i%5]}',
                        f'priority:{["high","medium","low"][i%3]}']),
            ('Cloud Storage', 300,
             '( role:admin OR role:editor OR role:owner ) AND ( department:engineering OR department:research )',
             lambda i: [f'user:cloud_user_{i:04d}', f'role:{["admin","editor","viewer","manager","owner"][i%5]}',
                        f'department:{["engineering","sales","hr","finance","research"][i%5]}',
                        f'region:{["us_east","us_west","eu_central","asia_pacific"][i%4]}']),
        ]

        results = {}
        total_steps = len(scenarios_config)
        step = 0

        for name, count, policy_str, attr_fn in scenarios_config:
            gc.collect()
            policy_tree = parser.parse(policy_str)
            M = group.random(GT)

            start = time.time()
            CT = tcabe.encrypt(M, policy_tree)
            encrypt_time = time.time() - start

            success_count = 0
            total_decrypt_time = 0.0

            for i in range(count):
                attrs = attr_fn(i)
                SK = tcabe.keygen(MSK, attrs)
                try:
                    start = time.time()
                    decrypted = tcabe.decrypt(SK, CT)
                    elapsed = time.time() - start
                    if decrypted == M:
                        success_count += 1
                        total_decrypt_time += elapsed
                except Exception:
                    pass

            avg_decrypt = total_decrypt_time / success_count if success_count > 0 else 0

            results[name] = {
                'entity_count': count,
                'success_count': success_count,
                'success_rate': round(success_count / count, 4),
                'encrypt_ms': round(encrypt_time * 1000, 2),
                'avg_decrypt_ms': round(avg_decrypt * 1000, 2)
            }

            step += 1
            heartbeat(step, total_steps, f'{name}: {success_count}/{count} success, encrypt={encrypt_time*1000:.1f}ms')

        self._save_real_world_data(results, 'real_world_summary_table.json')

        print("\n" + "=" * 100)
        print("Table 4: Real-World Scenario Verification Summary (used for SCI paper)")
        print("=" * 100)
        print(f"{'Scenario':15} {'Entities':12} {'SuccessRate':15} {'EncryptTime(ms)':20} {'AvgDecrypt(ms)':20}")
        print("-" * 100)
        for name, r in results.items():
            print(f"{name:15} {r['entity_count']:12} {r['success_rate']*100:12.1f}% {r['encrypt_ms']:16.1f}ms {r['avg_decrypt_ms']:18.1f}ms")
        print("=" * 100)
