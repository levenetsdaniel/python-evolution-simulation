"""Tests for neural/mutation.py"""

import numpy as np
import pytest

from neural.mutation import _build_features, neural_mutate
from neural.training_buffer import TrainingBuffer


N_GENES = 7
N_FEATURES = 22
MIN_TEMPERATURE = TrainingBuffer.MIN_TEMPERATURE
TEMP_RANGE = TrainingBuffer.MAX_TEMPERATURE - MIN_TEMPERATURE


@pytest.fixture
def advisor_factory():
    """Return a factory for trained advisor stubs with fixed predictions."""

    class AdvisorStub:
        is_trained = True

        def __init__(self, prediction):
            self._prediction = np.asarray(prediction, dtype=np.float32)

        def predict(self, x):
            return self._prediction.copy()

    return AdvisorStub


@pytest.fixture
def untrained_advisor():
    """Return an untrained advisor stub whose predict() must not be called."""

    class UntrainedAdvisorStub:
        is_trained = False

        def predict(self, x):
            pytest.fail("predict() must not be called for an untrained advisor")

    return UntrainedAdvisorStub()


def _features(**feature_overrides):
    """Build a feature vector with default values and optional overrides."""
    feature_args = dict(
        pre_mutation_genome=np.full(N_GENES, 0.5),
        env_params=[0.5, 1.0, 0.1],
        env_delta=[0.0, 0.0, 0.0],
        pop_mean_genome=np.full(N_GENES, 0.5),
        pop_mean_fitness=0.5,
        parent_mean_fitness=0.5,
    )
    feature_args.update(feature_overrides)
    return _build_features(**feature_args)


def _mutate(advisor, pre, shift=0.5, **mutate_kwargs):
    """Call neural_mutate() with stable default context values."""
    return neural_mutate(
        pre_mutation_genome=pre,
        advisor=advisor,
        env_params=[0.5, 1.0, 0.1],
        env_delta=[0.0, 0.0, 0.0],
        pop_mean_genome=np.full(N_GENES, 0.5),
        pop_mean_fitness=0.5,
        parent_mean_fitness=0.5,
        shift_strength=shift,
        **mutate_kwargs,
    )


def test_features_length():
    assert _features().shape == (N_FEATURES,)


def test_features_accept_dict_env():
    out = _features(
        env_params={
            "temperature": 20.0,
            "food_availability": 10000.0,
            "hazard_level": 0.1,
        },
        env_delta={
            "temperature": 0.0,
            "food_availability": 0.0,
            "hazard_level": 0.0,
        },
    )

    assert out.shape == (N_FEATURES,)
    assert out[N_GENES] == pytest.approx((20.0 - MIN_TEMPERATURE) / TEMP_RANGE)


def test_features_dtype_float32():
    assert _features().dtype == np.float32


def test_shift_math(advisor_factory):
    pre = np.full(N_GENES, 0.4, dtype=np.float32)
    advisor = advisor_factory(np.full(N_GENES, 0.8))

    mutated, deltas = _mutate(advisor, pre, shift=0.5)

    assert mutated == pytest.approx(np.full(N_GENES, 0.6), abs=1e-6)
    assert deltas == pytest.approx(np.full(N_GENES, 0.2), abs=1e-6)


def test_zero_shift_keeps_genome(advisor_factory):
    pre = np.full(N_GENES, 0.3, dtype=np.float32)
    advisor = advisor_factory(np.full(N_GENES, 0.9))

    mutated, deltas = _mutate(advisor, pre, shift=0.0)

    assert mutated == pytest.approx(pre)
    assert deltas == pytest.approx(np.zeros(N_GENES))


def test_full_shift_reaches_target(advisor_factory):
    pre = np.full(N_GENES, 0.3, dtype=np.float32)
    advisor = advisor_factory(np.full(N_GENES, 0.8))

    mutated, _ = _mutate(advisor, pre, shift=1.0)

    assert mutated == pytest.approx(np.full(N_GENES, 0.8))


def test_target_clipped_upper_bound(advisor_factory):
    pre = np.full(N_GENES, 0.5, dtype=np.float32)
    advisor = advisor_factory(np.full(N_GENES, 1.5))

    mutated, _ = _mutate(advisor, pre, shift=1.0)

    assert (mutated >= 0.0).all()
    assert (mutated <= 1.0).all()
    assert mutated == pytest.approx(np.ones(N_GENES))


def test_target_clipped_lower_bound(advisor_factory):
    pre = np.full(N_GENES, 0.5, dtype=np.float32)
    advisor = advisor_factory(np.full(N_GENES, -0.5))

    mutated, _ = _mutate(advisor, pre, shift=1.0)

    assert (mutated >= 0.0).all()
    assert (mutated <= 1.0).all()
    assert mutated == pytest.approx(np.zeros(N_GENES))


def test_deltas_match(advisor_factory):
    pre = np.full(N_GENES, 0.4, dtype=np.float32)
    advisor = advisor_factory(np.full(N_GENES, 0.8))

    mutated, deltas = _mutate(advisor, pre)

    assert deltas == pytest.approx(mutated - pre)


def test_wrong_target_shape(advisor_factory):
    pre = np.full(N_GENES, 0.5, dtype=np.float32)
    advisor = advisor_factory(np.full(N_GENES + 1, 0.8))

    with pytest.raises(ValueError, match="target shape"):
        _mutate(advisor, pre)


def test_fallback_changes_genome(untrained_advisor):
    pre = np.full(N_GENES, 0.5, dtype=np.float32)

    mutated, _ = _mutate(
        untrained_advisor,
        pre,
        rng=np.random.default_rng(0),
    )

    assert not np.allclose(mutated, pre)


def test_fallback_in_unit_interval(untrained_advisor):
    """Fallback mutation must keep the genome inside the [0.0, 1.0] interval."""
    pre = np.full(N_GENES, 0.5, dtype=np.float32)

    mutated, _ = _mutate(
        untrained_advisor,
        pre,
        mutation_std=1.0,
        rng=np.random.default_rng(0),
    )

    assert (mutated >= 0.0).all()
    assert (mutated <= 1.0).all()


def test_fallback_deltas_match(untrained_advisor):
    """Fallback deltas must equal mutated genome minus original genome."""
    pre = np.full(N_GENES, 0.5, dtype=np.float32)

    mutated, deltas = _mutate(
        untrained_advisor,
        pre,
        rng=np.random.default_rng(0),
    )

    assert deltas == pytest.approx(mutated - pre)


def test_fallback_works_without_rng(untrained_advisor):
    """Fallback mutation must work even when no RNG is provided."""
    pre = np.full(N_GENES, 0.5, dtype=np.float32)

    mutated, _ = _mutate(untrained_advisor, pre)

    assert mutated.shape == pre.shape
