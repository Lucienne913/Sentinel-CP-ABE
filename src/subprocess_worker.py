#!/usr/bin/env python3
"""
Subprocess Isolation Module: Core mechanism for OOM prevention

Reference OOM protection experience from Scheme 1, implements subprocess isolation for large matrix operations and intensive computations.

Design Principles:
1. Main process never directly executes operations that may cause OOM
2. Subprocess has strict memory limits and timeout
3. Results are passed via disk or IPC, without occupying main process memory
4. Subprocess crashes do not affect the main process

Architecture:
- Main process: Responsible for task distribution, result collection, error recovery
- Subprocess: Executes large matrix operations, pairing operation batch calculations
- Shared storage: Disk temporary files (avoids IPC memory copying)

Tech Stack:
- multiprocessing (process isolation)
- tempfile (disk temporary storage)
- psutil (memory monitoring)

References:
- Scheme 1 academic_implementation_v2 OOM protection experience
- Python multiprocessing best practices
"""

import multiprocessing as mp
import os
import tempfile
import time
import pickle
import traceback
from typing import Any, Dict, Optional, Tuple
from pathlib import Path


class SubprocessWorker:
    """
    Subprocess Worker: Safely execute operations that may cause OOM
    
    Use Cases:
    1. T-CP-ABE batch encryption/decryption
    2. Secret sharing for large policy trees
    3. Diffusion model batch generation
    
    Security Features:
    - Memory limit (MB): Subprocess is terminated if RSS exceeds threshold
    - Timeout limit (seconds): Operations are terminated on timeout
    - Exception isolation: Subprocess crashes do not affect main process
    - Result persistence: Results are first written to disk, then read back to main process
    """
    
    def __init__(self, max_memory_mb: int = 2048, timeout: int = 300):
        """
        Args:
            max_memory_mb: Maximum memory for subprocess (MB)
            timeout: Operation timeout (seconds)
        """
        self.max_memory_mb = max_memory_mb
        self.timeout = timeout
        self.temp_dir = Path(tempfile.gettempdir()) / 'scheme4_worker'
        self.temp_dir.mkdir(exist_ok=True)
        self._psutil_available = False
        try:
            import psutil
            self._psutil_available = True
        except ImportError:
            pass
    
    def _get_memory_usage_mb(self, pid: int = None) -> float:
        """Get process memory usage (MB)
        
        Args:
            pid: Process ID, None means current process
        """
        if not self._psutil_available:
            return 0.0
        try:
            import psutil
            process = psutil.Process(pid if pid else os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def execute_with_isolation(self, func, *args, timeout: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute function in isolated subprocess
        
        Args:
            func: Function to execute (must be picklable)
            *args: Function arguments
            timeout: Timeout in seconds, overrides default setting
            **kwargs: Function keyword arguments
            
        Returns:
            dict: {
                'success': bool,
                'result': Any,
                'error': str,
                'memory_usage_mb': float,
                'execution_time': float
            }
        """
        # Use specified timeout or default timeout
        actual_timeout = timeout if timeout is not None else self.timeout
        
        # Create temporary file for result transfer
        result_file = self.temp_dir / f'result_{os.getpid()}_{time.time()}.pkl'
        
        def _worker_process(result_path, func, args, kwargs, max_memory_mb):
            """Subprocess executes function and saves results to disk"""
            import threading
            
            memory_exceeded = [False]
            
            def memory_monitor():
                """Memory monitoring thread"""
                if not self._psutil_available:
                    return
                import psutil
                process = psutil.Process(os.getpid())
                while True:
                    try:
                        mem_usage = process.memory_info().rss / (1024 * 1024)
                        if mem_usage > max_memory_mb:
                            memory_exceeded[0] = True
                            # Set process self-termination flag
                            import signal
                            os.kill(os.getpid(), signal.SIGTERM)
                            break
                    except Exception:
                        break
                    time.sleep(0.1)
            
            # Start memory monitoring thread
            monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
            monitor_thread.start()
            
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                exec_time = time.time() - start_time
                
                # Check if terminated due to memory limit exceeded
                if memory_exceeded[0]:
                    raise MemoryError(f"Memory exceeded {max_memory_mb}MB")
                
                # Save results to disk
                with open(result_path, 'wb') as f:
                    pickle.dump({
                        'success': True,
                        'result': result,
                        'error': None,
                        'execution_time': exec_time,
                        'memory_usage_mb': self._get_memory_usage_mb(os.getpid())
                    }, f)
            except MemoryError as e:
                with open(result_path, 'wb') as f:
                    pickle.dump({
                        'success': False,
                        'result': None,
                        'error': f'MemoryError: {str(e)}',
                        'execution_time': 0.0,
                        'memory_usage_mb': max_memory_mb
                    }, f)
            except Exception as e:
                # Save error information
                with open(result_path, 'wb') as f:
                    pickle.dump({
                        'success': False,
                        'result': None,
                        'error': f'{type(e).__name__}: {str(e)}\n{traceback.format_exc()}',
                        'execution_time': 0.0,
                        'memory_usage_mb': self._get_memory_usage_mb(os.getpid())
                    }, f)
        
        # Start subprocess
        process = mp.Process(
            target=_worker_process,
            args=(str(result_file), func, args, kwargs, self.max_memory_mb)
        )
        process.start()
        
        # Wait for completion or timeout
        process.join(timeout=actual_timeout)
        
        # Check results
        if process.is_alive():
            # Timeout: force termination
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            
            return {
                'success': False,
                'result': None,
                'error': f'Timeout after {actual_timeout}s',
                'memory_usage_mb': 0.0,
                'execution_time': actual_timeout
            }
        
        # Read results
        if result_file.exists():
            try:
                with open(result_file, 'rb') as f:
                    result_data = pickle.load(f)
                return result_data
            except Exception as e:
                return {
                    'success': False,
                    'result': None,
                    'error': f'Result read error: {e}',
                    'memory_usage_mb': 0.0,
                    'execution_time': 0.0
                }
            finally:
                # Clean up temporary files
                try:
                    result_file.unlink()
                except Exception:
                    pass
        else:
            return {
                'success': False,
                'result': None,
                'error': f'Process exited with code {process.exitcode}',
                'memory_usage_mb': 0.0,
                'execution_time': 0.0
            }


def batch_encrypt_worker(PP, messages, policy_tree):
    """
    Batch encryption worker process
    
    Args:
        PP: Public parameters
        messages: Message list
        policy_tree: Access policy tree
        
    Returns:
        list: Ciphertext list
    """
    import sys
    from pathlib import Path
    # Ensure src directory is in sys.path
    src_dir = str(Path(__file__).parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from t_cp_abe import T_CP_ABE
    tcabe = T_CP_ABE(PP)
    return [tcabe.encrypt(M, policy_tree) for M in messages]


def batch_decrypt_worker(PP, SK, ciphertexts, current_time):
    """
    Batch decryption worker process
    
    Args:
        PP: Public parameters
        SK: User secret key
        ciphertexts: Ciphertext list
        current_time: Current time
        
    Returns:
        list: Decrypted message list
    """
    import sys
    from pathlib import Path
    # Ensure src directory is in sys.path
    src_dir = str(Path(__file__).parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from t_cp_abe import T_CP_ABE
    tcabe = T_CP_ABE(PP)
    results = []
    for CT in ciphertexts:
        try:
            M = tcabe.decrypt(SK, CT, current_time)
            results.append({'success': True, 'message': M})
        except ValueError as e:
            results.append({'success': False, 'error': str(e)})
    return results


def main():
    """Test subprocess isolation mechanism"""
    print("=" * 60)
    print("Scheme 4: Subprocess Isolation Mechanism Test")
    print("=" * 60)
    
    worker = SubprocessWorker(max_memory_mb=1024, timeout=60)
    
    # Test 1: Simple function execution
    print("\n[Test 1] Simple function execution")
    def simple_task(x, y):
        return x + y
    
    result = worker.execute_with_isolation(simple_task, 10, 20)
    print(f"  Result: {result}")
    assert result['success'] and result['result'] == 30
    
    # Test 2: Timeout detection
    print("\n[Test 2] Timeout detection")
    def slow_task():
        time.sleep(10)
        return "done"
    
    result = worker.execute_with_isolation(slow_task, timeout=2)
    print(f"  Timeout result: {result['error']}")
    assert not result['success'] and 'Timeout' in result['error']
    
    # Test 3: Exception isolation
    print("\n[Test 3] Exception isolation")
    def failing_task():
        raise ValueError("Intentional failure")
    
    result = worker.execute_with_isolation(failing_task)
    print(f"  Exception result: {result['error'][:50]}...")
    assert not result['success'] and 'ValueError' in result['error']
    
    print("\n" + "=" * 60)
    print("Subprocess Isolation Mechanism Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
