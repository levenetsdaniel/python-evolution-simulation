"""Tests for ``core/fitness.py``."""

import numpy as np
import pytest

from config.sim_config import EnvironmentConfig, FitnessConfig
from core.fitness import _energy_score, _hazard_score, _temp_score, _proportion_score, fitness

FCFG = FitnessConfig()
ECFG = EnvironmentConfig()
FLOOR = FCFG.score_floor


def _comfort_temp() -> float:
    """Real temperature at which ``temp_norm == temp_20_norm`` (i.e. delta == 0)."""
    return FCFG.temp_20_norm * (ECFG.max_temperature - ECFG.min_temperature) + ECFG.min_temperature


class TestTempScore:
    """Temperature adaptation score."""

    def test_perfect_score_when_temp_res_equals_optimum(self):
        s = _temp_score(0.5, 0.5, _comfort_temp(), optimum=1.0)
        assert s == pytest.approx(1.0)

    def test_at_comfort_temp_score_is_independent_of_genes(self):
        comfort = _comfort_temp()
        scores = [
            _temp_score(0.0, 0.0, comfort, optimum=0.5),
            _temp_score(1.0, 1.0, comfort, optimum=0.5),
            _temp_score(0.3, 0.7, comfort, optimum=0.5),
            _temp_score(0.9, 0.1, comfort, optimum=0.5),
        ]
        assert all(s == pytest.approx(scores[0]) for s in scores)

    def test_heat_resistance_helps_in_heat(self):
        hot = ECFG.max_temperature - 1.0
        good = _temp_score(heat_res=1.0, cold_res=0.5, temp=hot, optimum=1.0)
        bad = _temp_score(heat_res=0.0, cold_res=0.5, temp=hot, optimum=1.0)
        assert good > bad

    def test_cold_resistance_helps_in_cold(self):
        cold = ECFG.min_temperature + 1.0
        good = _temp_score(heat_res=0.5, cold_res=1.0, temp=cold, optimum=1.0)
        bad = _temp_score(heat_res=0.5, cold_res=0.0, temp=cold, optimum=1.0)
        assert good > bad

    def test_full_grid_within_bounds(self):
        for heat in np.linspace(0.0, 1.0, 5):
            for cold in np.linspace(0.0, 1.0, 5):
                for temp in np.linspace(ECFG.min_temperature, ECFG.max_temperature, 7):
                    for opt in np.linspace(0.0, 1.0, 5):
                        s = _temp_score(heat, cold, temp, opt)
                        assert FLOOR - 1e-9 <= s <= 1.0 + 1e-9


class TestHazardScore:
    """Hazard / fragility score."""

    @pytest.mark.parametrize("hazard", [0.0, 0.5, 1.0, 5.0, 100.0])
    def test_resilience_one_neutralizes_any_hazard(self, hazard):
        assert _hazard_score(resilience=1.0, hazard=hazard) == pytest.approx(1.0)

    @pytest.mark.parametrize("resilience", [0.0, 0.3, 0.7, 1.0])
    def test_zero_hazard_gives_perfect_score(self, resilience):
        assert _hazard_score(resilience=resilience, hazard=0.0) == pytest.approx(1.0)

    def test_monotonically_decreasing_in_hazard(self):
        scores = [_hazard_score(resilience=0.3, hazard=h)
                  for h in np.linspace(0.0, 3.0, 25)]
        for prev, cur in zip(scores, scores[1:]):
            assert cur <= prev + 1e-12

    def test_monotonically_increasing_in_resilience(self):
        scores = [_hazard_score(resilience=r, hazard=1.0)
                  for r in np.linspace(0.0, 1.0, 25)]
        for prev, cur in zip(scores, scores[1:]):
            assert cur >= prev - 1e-12

    def test_floor_clips_extreme_hazard(self):
        assert _hazard_score(resilience=0.0, hazard=10.0) == pytest.approx(FLOOR)

    def test_full_grid_within_bounds(self):
        for r in np.linspace(0.0, 1.0, 6):
            for h in np.linspace(0.0, 5.0, 8):
                s = _hazard_score(r, h)
                assert FLOOR - 1e-9 <= s <= 1.0 + 1e-9


class TestEnergyScore:
    """Energy / metabolic score."""

    def test_perfect_inputs_give_perfect_score(self):
        s = _energy_score(satiation=1.0, resilience=0.0,
                          metabolic_rate=0.0, aggressiveness=0.0)
        assert s == pytest.approx(1.0)

    def test_zero_satiation_clipped_to_floor(self):
        s = _energy_score(satiation=0.0, resilience=0.0,
                          metabolic_rate=0.0, aggressiveness=0.0)
        assert s == pytest.approx(FLOOR)

    def test_higher_satiation_gives_higher_score(self):
        low = _energy_score(satiation=0.2, resilience=0.0,
                            metabolic_rate=0.0, aggressiveness=0.0)
        high = _energy_score(satiation=0.9, resilience=0.0,
                             metabolic_rate=0.0, aggressiveness=0.0)
        assert high > low

    def test_aggression_above_metabolic_rate_adds_penalty(self):
        without_excess = _energy_score(satiation=1.0, resilience=0.0,
                                       metabolic_rate=0.3, aggressiveness=0.2)
        with_excess = _energy_score(satiation=1.0, resilience=0.0,
                                    metabolic_rate=0.3, aggressiveness=0.8)
        expected_penalty = (0.8 - 0.3) * FCFG.aggression_metabolic_penalty
        assert without_excess == pytest.approx(0.85)
        assert with_excess == pytest.approx(0.85 - expected_penalty)
        assert with_excess < without_excess

    def test_aggression_at_or_below_metabolic_rate_has_no_penalty(self):
        s_below = _energy_score(satiation=1.0, resilience=0.0,
                                metabolic_rate=0.5, aggressiveness=0.1)
        s_equal = _energy_score(satiation=1.0, resilience=0.0,
                                metabolic_rate=0.5, aggressiveness=0.5)
        assert s_below == pytest.approx(s_equal)

    def test_high_metabolic_and_resilience_reduce_efficiency(self):
        baseline = _energy_score(satiation=1.0, resilience=0.0,
                                 metabolic_rate=0.0, aggressiveness=0.0)
        costly = _energy_score(satiation=1.0, resilience=1.0,
                               metabolic_rate=1.0, aggressiveness=0.0)
        assert costly < baseline

    def test_full_grid_within_bounds(self):
        for sat in np.linspace(0.0, 1.5, 5):
            for met in np.linspace(0.0, 1.0, 4):
                for res in np.linspace(0.0, 1.0, 4):
                    for agg in np.linspace(0.0, 1.0, 4):
                        s = _energy_score(sat, res, met, agg)
                        assert FLOOR - 1e-9 <= s <= 1.0 + 1e-9


def _make_ind(**params) -> dict:
    base = dict(
        heat_resistance=0.5,
        cold_resistance=0.5,
        metabolic_rate=0.3,
        resilience=0.5,
        size=0.5,
        speed=0.5,
        aggressiveness=0.3,
        satiation=1.0,
    )
    base.update(params)
    return base


def _make_env(**params) -> dict:
    base = dict(
        temperature=_comfort_temp(),
        optimum_temperature=1.0,
        food_availability=10000.0,
        hazard_level=0.0,
    )
    base.update(params)
    return base


class TestProportionScore:
    """Body-proportion score."""

    def test_perfect_score_when_within_bounds(self):
        assert _proportion_score(speed=0.5, size=0.5, metabolic_rate=0.5) == pytest.approx(1.0)

    def test_perfect_score_at_exact_ratios(self):
        assert _proportion_score(speed=1.0, size=0.75, metabolic_rate=0.5) == pytest.approx(1.0)

    def test_speed_excess_lowers_score(self):
        baseline = _proportion_score(speed=0.4, size=0.0, metabolic_rate=0.2)
        bad = _proportion_score(speed=1.0, size=0.0, metabolic_rate=0.2)
        assert baseline == pytest.approx(1.0)
        assert bad == pytest.approx(0.4)

    def test_size_excess_lowers_score(self):
        bad = _proportion_score(speed=0.0, size=1.0, metabolic_rate=0.2)
        assert bad == pytest.approx(0.3)

    def test_clipped_to_score_floor_for_extreme_disproportion(self):
        s = _proportion_score(speed=1.0, size=1.0, metabolic_rate=0.0)
        assert s == pytest.approx(FLOOR)

    def test_zero_metabolic_with_zero_body_is_fine(self):
        assert _proportion_score(speed=0.0, size=0.0, metabolic_rate=0.0) == pytest.approx(1.0)

    def test_full_grid_within_bounds(self):
        for speed in np.linspace(0.0, 1.0, 5):
            for size in np.linspace(0.0, 1.0, 5):
                for met in np.linspace(0.0, 1.0, 5):
                    s = _proportion_score(speed, size, met)
                    assert FLOOR - 1e-9 <= s <= 1.0 + 1e-9


class TestFitness:
    """Composite fitness = temp_score * energy_score * hazard_score."""

    def test_equals_product_of_components(self):
        ind = _make_ind()
        env = _make_env()
        expected = (
                _temp_score(ind["heat_resistance"], ind["cold_resistance"],
                            env["temperature"], env["optimum_temperature"])
                * _energy_score(ind["satiation"], ind["resilience"],
                                ind["metabolic_rate"], ind["aggressiveness"])
                * _hazard_score(ind["resilience"], env["hazard_level"])
                * _proportion_score(ind["speed"], ind["size"], ind["metabolic_rate"])
        )
        assert fitness(ind, env) == pytest.approx(expected)

    def test_low_satiation_lowers_total_fitness(self):
        env = _make_env()
        starved = _make_ind(satiation=0.05)
        full = _make_ind(satiation=1.0)
        assert fitness(starved, env) < fitness(full, env)

    def test_high_resilience_wins_in_hazardous_environment(self):
        env = _make_env(hazard_level=1.0)
        ind_low = _make_ind(resilience=0.1)
        ind_high = _make_ind(resilience=0.9)
        assert fitness(ind_high, env) > fitness(ind_low, env)

    def test_returns_finite_value_in_unit_interval_for_random_inputs(self):
        rng = np.random.default_rng(42)
        for _ in range(200):
            ind = _make_ind(
                heat_resistance=float(rng.random()),
                cold_resistance=float(rng.random()),
                metabolic_rate=float(rng.random()),
                resilience=float(rng.random()),
                aggressiveness=float(rng.random()),
                satiation=float(rng.uniform(0.0, 1.5)),
            )
            env = _make_env(
                temperature=float(rng.uniform(ECFG.min_temperature, ECFG.max_temperature)),
                optimum_temperature=float(rng.random()),
                hazard_level=float(rng.uniform(0.0, 2.0)),
            )
            f = fitness(ind, env)
            assert 0.0 < f <= 1.0 + 1e-9
