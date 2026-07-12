"""
Scheme 4: Digital Twin Network Security Framework - Test Configuration

Features:
- Global fixtures (shared setup)
- Heartbeat mechanism (periodic progress output)
- Memory monitoring (psutil)
- OOM protection (subprocess isolation)
- Swap space monitoring
"""

import pytest
import os
import sys
import time
import gc
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import psutil (optional)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Try to import charm (required)
try:
    from charm.toolbox.pairinggroup import PairingGroup
    HAS_CHARM = True
except ImportError:
    HAS_CHARM = False


def pytest_runtest_logreport(report):
    """Test run report hook (heartbeat output)"""
    if report.when == 'call':
        elapsed = report.duration
        status = "PASS" if report.passed else "FAIL"
        print(f"\n  {status} {report.nodeid} ({elapsed:.2f}s)")


@pytest.fixture(scope="session")
def heartbeat():
    """
    Heartbeat mechanism: Periodically output test progress
    
    Usage:
    def test_something(heartbeat):
        for i in range(100):
            heartbeat(i, 100, "Processing")
            # Actual test logic
    """
    last_time = [time.time()]
    
    def beat(current, total, message=""):
        now = time.time()
        if now - last_time[0] >= 2.0:  # Output every 2 seconds
            pct = (current / total) * 100
            elapsed = now - last_time[0]
            print(f"\n    [HEARTBEAT] {current}/{total} ({pct:.1f}%) - {message} ({elapsed:.1f}s)")
            last_time[0] = now
    
    return beat


@pytest.fixture(scope="session")
def memory_monitor():
    """
    Memory monitor
    
    Usage:
    def test_memory_leak(memory_monitor):
        mem_before = memory_monitor.get_usage()
        # Test logic
        mem_after = memory_monitor.get_usage()
        assert mem_after - mem_before < threshold
    """
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid()) if HAS_PSUTIL else None
            self.snapshots = []
        
        def get_usage(self):
            """Get current memory usage (MB)"""
            if self.process:
                mem = self.process.memory_info().rss / (1024 * 1024)
                self.snapshots.append(('usage', mem))
                return mem
            return 0.0
        
        def get_swap_usage(self):
            """Get swap usage (MB)"""
            if HAS_PSUTIL:
                swap = psutil.swap_memory()
                swap_mb = swap.used / (1024 * 1024)
                self.snapshots.append(('swap', swap_mb))
                return swap_mb
            return 0.0
        
        def get_report(self):
            """Generate memory usage report"""
            if not self.snapshots:
                return "No memory data"
            
            usage_data = [s[1] for s in self.snapshots if s[0] == 'usage']
            swap_data = [s[1] for s in self.snapshots if s[0] == 'swap']
            
            report = []
            report.append("=== Memory Usage Report ===")
            if usage_data:
                report.append(f"  Memory usage: min={min(usage_data):.1f}MB, max={max(usage_data):.1f}MB, avg={sum(usage_data)/len(usage_data):.1f}MB")
            if swap_data:
                report.append(f"  Swap usage: min={min(swap_data):.1f}MB, max={max(swap_data):.1f}MB, avg={sum(swap_data)/len(swap_data):.1f}MB")
            return '\n'.join(report)
    
    return MemoryMonitor()


@pytest.fixture(scope="session")
def oom_protection():
    """
    OOM protection mechanism
    
    Features:
    - Memory usage monitoring
    - Force GC when threshold exceeded
    - Swap space monitoring
    - Subprocess isolation support
    """
    class OOMProtection:
        def __init__(self, threshold_mb=1024):
            self.threshold_mb = threshold_mb
            self.process = psutil.Process(os.getpid()) if HAS_PSUTIL else None
        
        def check_and_protect(self):
            """Check memory usage and protect if necessary"""
            if self.process:
                mem_mb = self.process.memory_info().rss / (1024 * 1024)
                if mem_mb > self.threshold_mb:
                    print(f"\n    ⚠ Memory usage {mem_mb:.1f}MB exceeds threshold {self.threshold_mb}MB, executing GC...")
                    gc.collect()
                    mem_after = self.process.memory_info().rss / (1024 * 1024)
                    print(f"    ✓ Memory after GC: {mem_after:.1f}MB")
                return mem_mb
            return 0.0
        
        def get_swap_status(self):
            """Get swap space status"""
            if HAS_PSUTIL:
                swap = psutil.swap_memory()
                return {
                    'total_gb': swap.total / (1024**3),
                    'used_mb': swap.used / (1024**2),
                    'free_mb': swap.free / (1024**2),
                    'percent': swap.percent
                }
            return None
    
    return OOMProtection(threshold_mb=2048)


@pytest.fixture(scope="session")
def system_setup():
    """System initialization fixture (shared by all tests)"""
    from src.setup import T_CP_ABE_Setup
    from src.t_cp_abe import T_CP_ABE, PolicyParser
    from src.auth import BidirectionalAuth
    from src.signatures import BLSSignature, DeviceCertificate
    from src.digital_twin import DigitalTwinManager
    from src.subprocess_worker import SubprocessWorker
    
    # Initialize system
    setup = T_CP_ABE_Setup(group_name='SS1024', security_level=128)
    PP, MSK = setup.setup(max_attrs=50, cache_max_size=1000)
    tcabe = T_CP_ABE(PP)
    auth = BidirectionalAuth(PP, tcabe=tcabe, max_nonce_cache_size=500, max_active_sessions=100)
    bls = BLSSignature(setup.group)
    cert_auth = DeviceCertificate(ca_sk=None, ca_pk=None, group=setup.group)
    dt_manager = DigitalTwinManager(use_local=True, max_history_size=100)
    worker = SubprocessWorker(max_memory_mb=1024, timeout=60)
    
    # Time predicate
    time_predicates = {
        'work': {'hour': (8, 18), 'weekday': [1, 2, 3, 4, 5]},
        'night': {'hour': (0, 6)}
    }
    parser = PolicyParser(time_predicates=time_predicates)
    
    return {
        'setup': setup,
        'PP': PP,
        'MSK': MSK,
        'tcabe': tcabe,
        'auth': auth,
        'bls': bls,
        'cert_auth': cert_auth,
        'dt_manager': dt_manager,
        'worker': worker,
        'parser': parser,
        'group': setup.group
    }
