from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import torch

MODULE_NAME = "modules.openclaw_generation_profile"
MODULE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modules", "openclaw_generation_profile.py")
MAX_SIZE_ENV = "OPENCLAW_GENERATION_PROFILE_CACHE_MAX"


def load_profile(testcase: unittest.TestCase, max_size: str = "16"):
    """Load a fresh copy of the module under its import-time size limit; the process' copy and environment come back after the test."""
    saved_env = os.environ.get(MAX_SIZE_ENV)
    saved_module = sys.modules.get(MODULE_NAME)

    def restore():
        if saved_env is None:
            os.environ.pop(MAX_SIZE_ENV, None)
        else:
            os.environ[MAX_SIZE_ENV] = saved_env
        if saved_module is None:
            sys.modules.pop(MODULE_NAME, None)
        else:
            sys.modules[MODULE_NAME] = saved_module

    testcase.addCleanup(restore)
    os.environ[MAX_SIZE_ENV] = max_size
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    profile = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = profile  # dataclasses resolve the module's string annotations through sys.modules
    spec.loader.exec_module(profile)
    profile.clear()
    return profile


def cache_sigmas(profile, steps, tensor, params):
    return profile.cached_tensor("sigmas", "Euler", "Karras", steps, tensor.device, tensor.dtype, lambda: tensor, params=params)


class GenerationProfileCacheTests(unittest.TestCase):
    def test_cached_tensor_reuses_exact_key(self):
        profile = load_profile(self)

        first = cache_sigmas(profile, 20, torch.arange(3), ("a",))
        second = cache_sigmas(profile, 20, torch.arange(3) + 10, ("a",))

        # A hit returns a private copy of the stored value, so a caller that edits its schedule cannot poison the cache.
        self.assertIsNot(second, first)
        self.assertEqual(profile.status()["hits"], 1)
        torch.testing.assert_close(second, torch.arange(3))
        first.fill_(-1)
        second.fill_(-2)
        torch.testing.assert_close(cache_sigmas(profile, 20, torch.arange(3) + 20, ("a",)), torch.arange(3))

    def test_cache_key_includes_params(self):
        profile = load_profile(self)

        first = cache_sigmas(profile, 20, torch.arange(3), ("a",))
        second = cache_sigmas(profile, 20, torch.arange(3) + 10, ("b",))

        self.assertIsNot(second, first)
        torch.testing.assert_close(second, torch.arange(3) + 10)

    def test_cached_tensor_skips_factory_on_hit(self):
        profile = load_profile(self)
        calls = []

        first = profile.cached_tensor(
            "sigmas",
            "Euler",
            "Karras",
            20,
            "cpu",
            torch.float32,
            lambda: calls.append("miss") or torch.arange(3, dtype=torch.float32),
            params=("a",),
        )
        second = profile.cached_tensor(
            "sigmas",
            "Euler",
            "Karras",
            20,
            "cpu",
            torch.float32,
            lambda: calls.append("hit") or torch.arange(3, dtype=torch.float32) + 10,
            params=("a",),
        )

        self.assertIsNot(second, first)
        torch.testing.assert_close(second, torch.arange(3, dtype=torch.float32))
        self.assertEqual(calls, ["miss"])
        self.assertEqual(profile.status()["hits"], 1)

    def test_cache_key_includes_device_and_dtype(self):
        profile = load_profile(self)

        first = profile.cached_tensor("sigmas", "Euler", "Karras", 20, "cpu", torch.float32, lambda: torch.ones(2))
        second = profile.cached_tensor("sigmas", "Euler", "Karras", 20, "cpu", torch.float64, lambda: torch.zeros(2, dtype=torch.float64))

        self.assertIsNot(second, first)
        self.assertEqual(second.dtype, torch.float64)
        self.assertEqual(profile.status()["cache_size"], 2)

    def test_cache_is_always_enabled_without_enable_env(self):
        profile = load_profile(self)
        key = profile.GenerationProfileKey("sigmas", "Euler", "Karras", 20, "cpu", "torch.float32")

        first = profile.tensor_for_key(key, lambda: torch.ones(2))
        second = profile.tensor_for_key(key, lambda: torch.zeros(2))

        self.assertTrue(profile.enabled())
        self.assertIsNot(second, first)
        torch.testing.assert_close(second, torch.ones(2))
        status = profile.status()
        self.assertTrue(status["enabled"])
        self.assertEqual(status["cache_size"], 1)
        self.assertEqual(status["hits"], 1)

    def test_max_size_zero_bypasses_without_store(self):
        profile = load_profile(self, max_size="0")
        key = profile.GenerationProfileKey("sigmas", "Euler", "Karras", 20, "cpu", "torch.float32")

        value = profile.tensor_for_key(key, lambda: torch.ones(2))

        torch.testing.assert_close(value, torch.ones(2))
        status = profile.status()
        self.assertEqual(status["cache_size"], 0)
        self.assertGreaterEqual(status["bypasses"], 1)
        self.assertEqual(status["last_bypass_reason"], "max_size_zero")

    def test_cache_honors_max_size(self):
        profile = load_profile(self, max_size="1")

        cache_sigmas(profile, 20, torch.arange(3), ("a",))
        cache_sigmas(profile, 21, torch.arange(4), ("a",))

        status = profile.status()
        self.assertEqual(status["cache_size"], 1)
        self.assertEqual(status["evictions"], 1)


if __name__ == "__main__":
    unittest.main()
