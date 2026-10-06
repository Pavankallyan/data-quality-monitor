"""Shared fixtures for data-quality-monitor tests."""

import numpy as np
import pandas as pd
import pytest

RNG = np.random.default_rng(42)


@pytest.fixture
def normal_base():
    return pd.Series(RNG.normal(100, 15, 5_000), name="temperature_c")


@pytest.fixture
def normal_shifted():
    return pd.Series(RNG.normal(120, 15, 5_000), name="temperature_c")


@pytest.fixture
def categorical_base():
    return pd.Series(
        RNG.choice(["OK", "WARN", "FAIL"], 2_000, p=[0.8, 0.15, 0.05]),
        name="status",
    )


@pytest.fixture
def categorical_shifted():
    return pd.Series(
        RNG.choice(["OK", "WARN", "FAIL"], 2_000, p=[0.3, 0.4, 0.3]),
        name="status",
    )
