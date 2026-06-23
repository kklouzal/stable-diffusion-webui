from __future__ import annotations

import importlib
import os
import sys
import types
import unittest

import torch


def load_profile(max_size: str = "16"):
    os.environ.pop("OPENCLAW_GENERATION_PROFILE_CACHE", None)
    os.environ["OPENCLAW_GENERATION_PROFILE_CACHE_MAX"] = max_size
    sys.modules.pop("modules.openclaw_generation_profile", None)
    modules_pkg = sys.modules.setdefault("modules", types.ModuleType("modules"))
    modules_pkg.__path__ = ["modules"]
    profile = importlib.import_module("modules.openclaw_generation_profile")
    profile.clear()
    return profile


class GenerationProfileCacheTests(unittest.TestCase):
    def test_cache_tensor_reuses_exact_key(self):
        profile = load_profile()

        first = profile.cache_tensor("sigmas", "Euler", "Karras", 20, torch.arange(3), params=("a",))
        second = profile.cache_tensor("sigmas", "Euler", "Karras", 20, torch.arange(3) + 10, params=("a",))

        self.assertIs(second, first)
        self.assertEqual(profile.status()["hits"], 1)
        torch.testing.assert_close(second, torch.arange(3))

    def test_cache_key_includes_params(self):
        profile = load_profile()

        first = profile.cache_tensor("sigmas", "Euler", "Karras", 20, torch.arange(3), params=("a",))
        second = profile.cache_tensor("sigmas", "Euler", "Karras", 20, torch.arange(3) + 10, params=("b",))

        self.assertIsNot(second, first)
        torch.testing.assert_close(second, torch.arange(3) + 10)

    def test_cached_tensor_skips_factory_on_hit(self):
        profile = load_profile()
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

        self.assertIs(second, first)
        self.assertEqual(calls, ["miss"])
        self.assertEqual(profile.status()["hits"], 1)

    def test_cache_key_includes_device_and_dtype(self):
        profile = load_profile()

        first = profile.cached_tensor("sigmas", "Euler", "Karras", 20, "cpu", torch.float32, lambda: torch.ones(2))
        second = profile.cached_tensor("sigmas", "Euler", "Karras", 20, "cpu", torch.float64, lambda: torch.zeros(2, dtype=torch.float64))

        self.assertIsNot(second, first)
        self.assertEqual(second.dtype, torch.float64)
        self.assertEqual(profile.status()["cache_size"], 2)

    def test_cache_is_always_enabled_without_enable_env(self):
        profile = load_profile()
        key = profile.GenerationProfileKey("sigmas", "Euler", "Karras", 20, "cpu", "torch.float32")

        first = profile.tensor_for_key(key, lambda: torch.ones(2))
        second = profile.tensor_for_key(key, lambda: torch.zeros(2))

        self.assertTrue(profile.enabled())
        self.assertIs(second, first)
        status = profile.status()
        self.assertTrue(status["enabled"])
        self.assertEqual(status["cache_size"], 1)
        self.assertEqual(status["hits"], 1)

    def test_max_size_zero_bypasses_without_store(self):
        profile = load_profile(max_size="0")
        key = profile.GenerationProfileKey("sigmas", "Euler", "Karras", 20, "cpu", "torch.float32")

        value = profile.tensor_for_key(key, lambda: torch.ones(2))

        torch.testing.assert_close(value, torch.ones(2))
        status = profile.status()
        self.assertEqual(status["cache_size"], 0)
        self.assertGreaterEqual(status["bypasses"], 1)
        self.assertEqual(status["last_bypass_reason"], "max_size_zero")

    def test_cache_honors_max_size(self):
        profile = load_profile(max_size="1")

        profile.cache_tensor("sigmas", "Euler", "Karras", 20, torch.arange(3), params=("a",))
        profile.cache_tensor("sigmas", "Euler", "Karras", 21, torch.arange(4), params=("a",))

        status = profile.status()
        self.assertEqual(status["cache_size"], 1)
        self.assertEqual(status["evictions"], 1)


if __name__ == "__main__":
    unittest.main()
