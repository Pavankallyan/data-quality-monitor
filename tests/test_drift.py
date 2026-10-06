"""Unit tests for src/drift.py — PSI, band labels, KS/chi-square drift tests,
column-level reports, and the schema diff / rename heuristic."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drift import (
    psi_band,
    psi_categorical,
    psi_numeric,
    chisq_drift,
    column_drift,
    ks_drift,
    schema_diff,
)


def test_psi_numeric_identical_is_zero(normal_base):
    assert psi_numeric(normal_base, normal_base) == pytest.approx(0.0, abs=1e-9)


def test_psi_numeric_shifted_is_significant(normal_base, normal_shifted):
    assert psi_numeric(normal_base, normal_shifted) > 0.25


def test_psi_numeric_empty_returns_zero():
    assert psi_numeric(pd.Series([], dtype=float), pd.Series([1.0])) == 0.0


def test_psi_numeric_constant_returns_zero():
    base = pd.Series([5.0] * 100)
    assert psi_numeric(base, base) == 0.0


def test_psi_categorical_identical_is_zero(categorical_base):
    assert psi_categorical(categorical_base, categorical_base) == pytest.approx(0.0, abs=1e-9)


def test_psi_categorical_shifted_is_significant(categorical_base, categorical_shifted):
    assert psi_categorical(categorical_base, categorical_shifted) > 0.25


def test_psi_band_boundaries():
    assert psi_band(0.0) == "no change"
    assert psi_band(0.0999) == "no change"
    assert psi_band(0.10) == "moderate"
    assert psi_band(0.25) == "moderate"
    assert psi_band(0.2501) == "significant"


def test_ks_drift_detects_shift(normal_base, normal_shifted):
    res = ks_drift(normal_base, normal_shifted)
    assert res["ks_stat"] > 0
    assert res["p_value"] < 0.05
    assert res["drifted"] is True


def test_ks_drift_no_shift_not_drifted(normal_base):
    res = ks_drift(normal_base, normal_base)
    assert res["drifted"] is False


def test_ks_drift_empty_returns_nulls():
    res = ks_drift(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert res["ks_stat"] is None and res["drifted"] is False


def test_chisq_drift_detects_shift(categorical_base, categorical_shifted):
    res = chisq_drift(categorical_base, categorical_shifted)
    assert res["chi2_stat"] > 0
    assert res["p_value"] < 0.05
    assert res["drifted"] is True


def test_chisq_drift_single_category_returns_nulls():
    s = pd.Series(["OK"] * 50)
    res = chisq_drift(s, s)
    assert res["chi2_stat"] is None and res["drifted"] is False


def test_column_drift_numeric_schema(normal_base, normal_shifted):
    df = pd.DataFrame({"temperature_c": normal_base, "x": normal_base})
    res = column_drift(df, pd.DataFrame({"temperature_c": normal_shifted, "x": normal_shifted}),
                       "temperature_c")
    assert res["type"] == "numeric"
    assert res["psi_band"] == "significant"
    assert set(res) >= {"column", "psi", "psi_band", "ks_stat", "p_value", "drifted"}


def test_column_drift_categorical_type(normal_base, categorical_base):
    df = pd.DataFrame({"status": categorical_base})
    res = column_drift(df, df, "status")
    assert res["type"] == "categorical"
    assert res["drifted"] is False


def test_schema_diff_added_removed():
    base = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    new = pd.DataFrame({"a": [1, 2], "c": [5, 6]})
    diff = schema_diff(base, new)
    assert diff["added"] == ["c"]
    assert diff["removed"] == ["b"]
    assert diff["renamed"] == []


def test_schema_diff_detects_rename():
    base = pd.DataFrame({"temp": ["a", "b", "c", "d"] * 25})
    new = pd.DataFrame({"temperature": ["a", "b", "c", "d"] * 25})
    diff = schema_diff(base, new)
    assert len(diff["renamed"]) == 1
    assert diff["renamed"][0]["from"] == "temp"
    assert diff["renamed"][0]["to"] == "temperature"
    assert diff["added"] == [] and diff["removed"] == []


def test_schema_diff_dtype_change():
    base = pd.DataFrame({"v": pd.Series([1, 2, 3], dtype="int64")})
    new = pd.DataFrame({"v": pd.Series(["1", "2", "3"], dtype="object")})
    diff = schema_diff(base, new)
    assert len(diff["dtype_changes"]) == 1
    assert diff["dtype_changes"][0]["column"] == "v"
    assert diff["dtype_changes"][0]["baseline_dtype"] == "int64"


def test_schema_diff_no_changes():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    diff = schema_diff(df, df.copy())
    assert diff["added"] == [] and diff["removed"] == []
    assert diff["renamed"] == [] and diff["dtype_changes"] == []
