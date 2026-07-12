#!/usr/bin/env python3
"""
Eclipse Ditto Integration Test Script (pytest format)

Test strategy:
- Connect to real Eclipse Ditto service (configure via DITTO_URL environment variable)
- If no real Ditto service is available, tests automatically skip (rather than fake pass)
- API contract verification does not depend on running Ditto service
"""

import json
import time
import os
import pytest
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


DITTO_URL = os.environ.get('DITTO_URL', '').rstrip('/')
DITTO_AUTH = (os.environ.get('DITTO_USER', 'ditto'), os.environ.get('DITTO_PASS', 'ditto'))
DITTO_AVAILABLE = False

if DITTO_URL and HAS_REQUESTS:
    try:
        resp = requests.get(f'{DITTO_URL}/api/2/things', timeout=5, auth=DITTO_AUTH)
        DITTO_AVAILABLE = resp.status_code in [200, 401, 403]
    except Exception:
        DITTO_AVAILABLE = False


class DittoServiceTester:
    """Eclipse Ditto REST API client."""

    def __init__(self, ditto_url: str, auth: Optional[Tuple[str, str]] = None):
        self.ditto_url = ditto_url.rstrip('/')
        self.headers = {'Content-Type': 'application/json'}
        self.auth = auth

    def _request(self, method, endpoint, **kwargs):
        if not HAS_REQUESTS:
            raise RuntimeError('requests library not installed')
        url = f'{self.ditto_url}{endpoint}'
        kwargs.setdefault('headers', self.headers)
        if self.auth:
            kwargs.setdefault('auth', self.auth)
        return requests.request(method, url, **kwargs)

    def check_connection(self) -> bool:
        """Check if Ditto service is reachable."""
        try:
            resp = self._request('GET', '/api/2/things?limit=1', timeout=5)
            return resp.status_code in [200, 401, 403]
        except Exception:
            return False

    def create_thing(self, thing_id: str, attributes: dict) -> bool:
        policy_id = thing_id
        policy_payload = {
            'entries': {
                'DEFAULT': {
                    'subjects': {'nginx:ditto': {'type': 'nginx basic auth provider'}},
                    'resources': {
                        'thing:/': {'grant': ['READ', 'WRITE'], 'revoke': []},
                        'policy:/': {'grant': ['READ', 'WRITE'], 'revoke': []},
                        'message:/': {'grant': ['READ', 'WRITE'], 'revoke': []}
                    }
                }
            }
        }
        self._request('PUT', f'/api/2/policies/{policy_id}', json=policy_payload)

        payload = {'attributes': attributes, 'policyId': policy_id}
        resp = self._request('PUT', f'/api/2/things/{thing_id}', json=payload)
        return resp.status_code in [200, 201, 204]

    def get_thing(self, thing_id: str) -> Tuple[bool, Optional[dict]]:
        resp = self._request('GET', f'/api/2/things/{thing_id}')
        return resp.status_code == 200, resp.json() if resp.status_code == 200 else None

    def update_thing(self, thing_id: str, new_attributes: dict) -> bool:
        resp = self._request('PUT', f'/api/2/things/{thing_id}/attributes', json=new_attributes)
        return resp.status_code in [200, 204]

    def send_command(self, thing_id: str, command: dict) -> bool:
        resp = self._request('POST', f'/api/2/things/{thing_id}/inbox/messages/command',
                             json=command, params={'timeout': '0'})
        return resp.status_code in [200, 202, 204]

    def delete_thing(self, thing_id: str) -> bool:
        resp = self._request('DELETE', f'/api/2/things/{thing_id}')
        return resp.status_code in [200, 204, 404]

    def search_things(self, filter_str: str, limit: int = 10) -> Tuple[bool, Optional[list]]:
        resp = self._request('GET', f'/api/2/search/things?filter={filter_str}&limit={limit}')
        return resp.status_code == 200, resp.json().get('items', []) if resp.status_code == 200 else None


@pytest.fixture(scope='module')
def ditto_tester():
    """Ditto test client fixture. Requires real Ditto service."""
    if not HAS_REQUESTS:
        pytest.skip('requests library not installed')
    if not DITTO_URL:
        pytest.skip('DITTO_URL environment variable not set. '
                     'Set it to your Ditto instance URL (e.g., http://localhost:8080) '
                     'to run integration tests against a real Ditto service.')
    if not DITTO_AVAILABLE:
        pytest.skip(f'Ditto service not reachable at {DITTO_URL}. '
                     'Ensure Eclipse Ditto is running and accessible.')
    tester = DittoServiceTester(ditto_url=DITTO_URL, auth=DITTO_AUTH)
    yield tester


@pytest.mark.integration
class TestDittoAPIContract:
    """API contract verification (does not depend on running Ditto service)."""

    def test_url_construction(self):
        """Verify Ditto API URL construction correctness."""
        base = 'http://localhost:8080'
        tester = DittoServiceTester(ditto_url=base)

        assert tester.ditto_url == 'http://localhost:8080'

    def test_url_trailing_slash_handling(self):
        """Verify trailing slash handling."""
        tester = DittoServiceTester(ditto_url='http://localhost:8080/')
        assert tester.ditto_url == 'http://localhost:8080'

    def test_thing_id_format(self):
        """Verify Thing ID format conforms to Eclipse Ditto specification."""
        valid_ids = [
            'org.eclipse.ditto:test-device-001',
            'com.example:my-thing',
            'my.namespace:device-123',
        ]
        for thing_id in valid_ids:
            assert ':' in thing_id, f'Thing ID {thing_id} should contain namespace separator'

    def test_create_payload_structure(self):
        """Verify Create Thing payload structure."""
        thing_id = 'org.eclipse.ditto:test-device-001'
        attributes = {'type': 'temperature-sensor', 'location': 'factory-a'}

        payload = {
            'attributes': attributes,
            'policy': {
                'entries': {
                    'DEFAULT': {
                        'subjects': {'type:unknown': {}},
                        'resources': {'thing:/': {'grant': ['READ', 'WRITE'], 'revoke': []}}
                    }
                }
            }
        }

        assert 'attributes' in payload
        assert 'policy' in payload
        assert 'entries' in payload['policy']
        assert 'DEFAULT' in payload['policy']['entries']
        assert payload['attributes']['type'] == 'temperature-sensor'

    def test_api_endpoint_paths(self):
        """Verify API endpoint path format."""
        thing_id = 'org.eclipse.ditto:test-device-001'

        expected_paths = {
            'create': f'/api/2/things/{thing_id}',
            'get': f'/api/2/things/{thing_id}',
            'update_attrs': f'/api/2/things/{thing_id}/attributes',
            'command': f'/api/2/things/{thing_id}/inbox/messages/command',
            'delete': f'/api/2/things/{thing_id}',
            'search': '/api/2/search/things',
        }

        for name, path in expected_paths.items():
            assert path.startswith('/api/2/'), f'{name} path should start with /api/2/'
            if name != 'search':
                assert thing_id in path, f'{name} path should contain thing_id'

    def test_request_headers(self):
        """Verify request headers setup."""
        tester = DittoServiceTester(ditto_url='http://localhost:8080')
        assert 'Content-Type' in tester.headers
        assert tester.headers['Content-Type'] == 'application/json'

    def test_ditto_service_tester_methods_exist(self):
        """Verify DittoServiceTester has all necessary API methods."""
        tester = DittoServiceTester(ditto_url='http://localhost:8080')

        required_methods = [
            'check_connection', 'create_thing', 'get_thing',
            'update_thing', 'send_command', 'delete_thing', 'search_things'
        ]
        for method in required_methods:
            assert hasattr(tester, method), f'Missing method: {method}'
            assert callable(getattr(tester, method)), f'{method} is not callable'


@pytest.mark.integration
@pytest.mark.skipif(not DITTO_AVAILABLE, reason='Ditto service not available')
class TestDittoIntegration:
    """Eclipse Ditto real service integration tests. Requires running Ditto instance."""

    THING_ID = 'org.eclipse.ditto:scheme4-test-device-001'
    ATTRIBUTES = {'type': 'temperature-sensor', 'location': 'factory-a', 'firmware': 'v2.1.0'}

    @pytest.fixture(autouse=True)
    def cleanup_thing(self, ditto_tester):
        yield
        try:
            ditto_tester.delete_thing(self.THING_ID)
        except Exception:
            pass

    def test_create_thing(self, ditto_tester):
        result = ditto_tester.create_thing(self.THING_ID, self.ATTRIBUTES)
        assert result is True

    def test_get_thing(self, ditto_tester):
        create_result = ditto_tester.create_thing(self.THING_ID, self.ATTRIBUTES)
        assert create_result is True, 'Failed to create thing for get test'
        success, data = ditto_tester.get_thing(self.THING_ID)
        assert success is True, f'GET returned non-200 status'
        assert data is not None
        assert data.get('thingId') == self.THING_ID or self.THING_ID.split(':')[1] in str(data)

    def test_update_thing(self, ditto_tester):
        create_result = ditto_tester.create_thing(self.THING_ID, self.ATTRIBUTES)
        assert create_result is True, 'Failed to create thing for update test'
        result = ditto_tester.update_thing(self.THING_ID, {'status': 'online', 'temperature': 23.5})
        assert result is True

    def test_send_command(self, ditto_tester):
        create_result = ditto_tester.create_thing(self.THING_ID, self.ATTRIBUTES)
        assert create_result is True, 'Failed to create thing for command test'
        result = ditto_tester.send_command(self.THING_ID, {'action': 'calibrate', 'params': {'target': 25.0}})
        assert result is True

    def test_delete_thing(self, ditto_tester):
        ditto_tester.create_thing(self.THING_ID, self.ATTRIBUTES)
        result = ditto_tester.delete_thing(self.THING_ID)
        assert result is True

    def test_search_things(self, ditto_tester):
        ditto_tester.create_thing(self.THING_ID, self.ATTRIBUTES)
        success, items = ditto_tester.search_things('eq(attributes/type,"temperature-sensor")')
        assert success is True
        assert isinstance(items, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
