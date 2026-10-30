import os
import sys
import unittest
import asyncio

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api.health import get_health
from app.api.schemas import HealthResponse
from app.models.model_manager import get_model_manager


class TestHealthEndpoint(unittest.TestCase):
    """Test Health Check logic directly."""

    def test_health_check_direct(self):
        """Test direct health endpoint invocation."""
        manager = get_model_manager()
        manager.mock_mode = True

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response: HealthResponse = loop.run_until_complete(get_health())
            self.assertEqual(response.status, "healthy")
            self.assertEqual(response.service, "SAGE API")
            self.assertIn(response.device, ["mock", "cuda", "cpu"])
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()

