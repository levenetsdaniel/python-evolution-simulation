"""Tests for neural/trainer.py"""

import numpy as np
import pytest

from neural.trainer import train_advisor
from neural.training_buffer import TrainingBuffer


N_GENES = 7
N_FEATURES = 22
MAX_FOOD = TrainingBuffer.MAX_FOOD


@pytest.fixture
def advisor_spy():
    """Return an advisor test double that records fit() arguments."""

    class AdvisorSpy:
        def __init__(self):
            self.is_trained = False
            self.fit_called = False
            self.X = None
            self.y = None
            self.weights = None

        def fit(self, X, y, weights=None):
            self.fit_called = True
            self.is_trained = True
            self.X = X
            self.y = y
            self.weights = weights

    return AdvisorSpy()


def _fill_buffer(buf, n=15, *, pre=None, deltas=None):
    """Fill a TrainingBuffer with deterministic birth and fitness records."""
    for i in range(n):
        buf.record_birth(
            child_id=i,
            pre_mutation_genome=pre if pre is not None else np.full(N_GENES, 0.5),
            mutation_deltas=deltas if deltas is not None else np.zeros(N_GENES),
            p1_fitness=0.4,
            p2_fitness=0.6,
            env_params={
                "temperature": 20.0,
                "food_availability": MAX_FOOD,
                "hazard_level": 0.1,
            },
            env_delta={
                "temperature": 0.0,
                "food_availability": 0.0,
                "hazard_level": 0.0,
            },
            pop_mean_genome=np.full(N_GENES, 0.5),
            pop_mean_fitness=0.5,
        )
        buf.record_fitness(i, 0.5 + 0.1 * (i % 3))


def test_empty_buffer_raises(advisor_spy):
    """train_advisor() must reject an empty TrainingBuffer."""
    with pytest.raises(ValueError):
        train_advisor(TrainingBuffer(), advisor_spy)


def test_returns_three_arrays(advisor_spy):
    """train_advisor() must return feature, target, and weight arrays."""
    buf = TrainingBuffer()
    _fill_buffer(buf)

    X, y, w = train_advisor(buf, advisor_spy)

    assert X is not None
    assert y is not None
    assert w is not None


def test_shapes_match(advisor_spy):
    """Returned arrays must have shapes matching the buffered samples."""
    buf = TrainingBuffer()
    _fill_buffer(buf, n=15)

    X, y, w = train_advisor(buf, advisor_spy)

    assert X.shape == (15, N_FEATURES)
    assert y.shape == (15, N_GENES)
    assert w.shape == (15,)


def test_calls_advisor_fit(advisor_spy):
    """train_advisor() must call advisor.fit() with prepared arrays."""
    buf = TrainingBuffer()
    _fill_buffer(buf)

    X, y, w = train_advisor(buf, advisor_spy)

    assert advisor_spy.fit_called is True
    assert advisor_spy.X is X
    assert advisor_spy.y is y
    assert advisor_spy.weights is w


def test_marks_trained(advisor_spy):
    """train_advisor() must mark the advisor as trained through fit()."""
    buf = TrainingBuffer()
    _fill_buffer(buf)

    train_advisor(buf, advisor_spy)

    assert advisor_spy.is_trained is True


def test_target_equals_pre_plus_deltas(advisor_spy):
    """Targets must be computed as pre-mutation genome plus mutation deltas."""
    buf = TrainingBuffer()
    _fill_buffer(
        buf,
        n=10,
        pre=np.full(N_GENES, 0.4),
        deltas=np.full(N_GENES, 0.1),
    )

    _, y, _ = train_advisor(buf, advisor_spy)

    assert y[0] == pytest.approx(np.full(N_GENES, 0.5))


def test_target_clipped_to_unit_interval(advisor_spy):
    """Targets must be clipped to the valid genome range [0.0, 1.0]."""
    buf = TrainingBuffer()
    _fill_buffer(
        buf,
        n=10,
        pre=np.full(N_GENES, 0.9),
        deltas=np.full(N_GENES, 0.5),
    )

    _, y, _ = train_advisor(buf, advisor_spy)

    assert (y >= 0.0).all()
    assert (y <= 1.0).all()
    assert y[0] == pytest.approx(np.ones(N_GENES))


def test_weights_sum_to_one(advisor_spy):
    """Returned sample weights must be normalized to sum to one."""
    buf = TrainingBuffer()
    _fill_buffer(buf, n=12)

    _, _, w = train_advisor(buf, advisor_spy)

    assert w.sum() == pytest.approx(1.0)
