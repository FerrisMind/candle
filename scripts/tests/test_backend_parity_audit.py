#!/usr/bin/env python3
"""Negative self-tests for scripts/backend_parity_audit.py."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = ROOT / "scripts" / "backend_parity_audit.py"


def load_audit():
    spec = importlib.util.spec_from_file_location("backend_parity_audit", AUDIT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Required so @dataclass can resolve the module namespace.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class AuditSelfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_audit()

    def test_immediate_unsupported_positive(self):
        body = 'fn foo(&self) -> Result<Self> {\n    return Err(unsupported("x"));\n}'
        self.assertTrue(self.m.is_immediately_unsupported(body))

    def test_immediate_unsupported_negative_gpu(self):
        body = (
            "fn foo(&self, layout: &Layout) -> Result<Self> {\n"
            "    self.run_unary_generic(layout, spirv)\n"
            "}"
        )
        self.assertFalse(self.m.is_immediately_unsupported(body))

    def test_usable_rejects_empty_error_body(self):
        info = self.m.analyze_method(
            "affine",
            1,
            'fn affine() { return Err(unsupported("a")); }',
        )
        self.assertFalse(self.m.usable(info))

    def test_usable_accepts_run_path(self):
        info = self.m.analyze_method(
            "affine",
            1,
            "fn affine(&self, l: &Layout, a: f64, b: f64) -> Result<Self> {\n"
            "    self.run_unary_generic_with_params(l, spirv, a, b)\n}",
        )
        self.assertTrue(self.m.usable(info))

    def test_parse_test_ref_with_function(self):
        f, fn = self.m.parse_test_ref(
            "candle-core/tests/gpu_parity_matrix_tests.rs::parity_special_floats"
        )
        self.assertEqual(fn, "parity_special_floats")
        self.assertIn("gpu_parity_matrix_tests.rs", f)

    def test_parse_test_ref_bare_file(self):
        f, fn = self.m.parse_test_ref("candle-core/tests/backend_smoke_tests.rs")
        self.assertIsNone(fn)
        self.assertIn("backend_smoke_tests.rs", f)

    def test_builtin_self_test_entry(self):
        self.assertEqual(self.m.run_self_tests(), 0)

    def test_real_repo_audit_runs(self):
        # May fail if manifest incomplete — only assert it returns int.
        rc = self.m.run_audit(ROOT, None, as_json=False)
        self.assertIn(rc, (0, 1))


if __name__ == "__main__":
    # Ensure failures surface as non-zero
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AuditSelfTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
