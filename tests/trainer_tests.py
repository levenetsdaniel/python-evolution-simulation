"""Tests for neural/trainer.py."""

import numpy as np
import pytest

from core.training_buffer import TrainingBuffer
from neural.advisor import CatBoostAdvisor
from neural.trainer import train_advisor


N_GENES = 7
N_FEATURES = 22
MAX_FOOD = TrainingBuffer.MAX_FOOD


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


def test_empty_buffer_raises():
    advisor = CatBoostAdvisor(iterations=10, verbose=False)

    with pytest.raises(ValueError):
        train_advisor(TrainingBuffer(), advisor)


def test_returns_three_arrays():
    buf = TrainingBuffer()
    _fill_buffer(buf)

    X, y, w = train_advisor(buf, CatBoostAdvisor(iterations=10, verbose=False))

    assert X is not None
    assert y is not None
    assert w is not None


def test_shapes_match():
    buf = TrainingBuffer()
    _fill_buffer(buf, n=15)

    X, y, w = train_advisor(buf, CatBoostAdvisor(iterations=10, verbose=False))

    assert X.shape == (15, N_FEATURES)
    assert y.shape == (15, N_GENES)
    assert w.shape == (15,)


def test_marks_trained():
    buf = TrainingBuffer()
    _fill_buffer(buf)

    advisor = CatBoostAdvisor(iterations=10, verbose=False)
    train_advisor(buf, advisor)

    assert advisor.is_trained is True


def test_target_equals_pre_plus_deltas():
    buf = TrainingBuffer()
    _fill_buffer(
        buf,
        n=10,
        pre=np.full(N_GENES, 0.4),
        deltas=np.full(N_GENES, 0.1),
    )

    _, y, _ = train_advisor(buf, CatBoostAdvisor(iterations=10, verbose=False))

    assert y[0] == pytest.approx(np.full(N_GENES, 0.5))


def test_target_clipped_to_unit_interval():
    buf = TrainingBuffer()
    _fill_buffer(
        buf,
        n=10,
        pre=np.full(N_GENES, 0.9),
        deltas=np.full(N_GENES, 0.5),
    )

    _, y, _ = train_advisor(buf, CatBoostAdvisor(iterations=10, verbose=False))

    assert (y >= 0.0).all()
    assert (y <= 1.0).all()
    assert y[0] == pytest.approx(np.ones(N_GENES))


def test_weights_sum_to_one():
    buf = TrainingBuffer()
    _fill_buffer(buf, n=12)

    _, _, w = train_advisor(buf, CatBoostAdvisor(iterations=10, verbose=False))

    assert w.sum() == pytest.approx(1.0)