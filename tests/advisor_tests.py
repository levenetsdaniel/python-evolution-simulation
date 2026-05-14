import numpy as np
import pytest

from neural.advisor import CatBoostAdvisor


N_FEATS = 22
N_GENES = 7


def _data(n_samples: int = 30, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, N_FEATS)).astype(np.float32)
    y = rng.random((n_samples, N_GENES)).astype(np.float32)
    weights = np.ones(n_samples, dtype=np.float32) / n_samples
    return X, y, weights


@pytest.fixture(scope="module")
def advisor_fit():
    X, y, weights = _data()
    advisor = CatBoostAdvisor(iterations=10, verbose=False)
    advisor.fit(X, y, weights)
    return advisor


def test_new_is_untrained():
    advisor = CatBoostAdvisor()
    assert advisor.is_trained is False


def test_new_has_model():
    advisor = CatBoostAdvisor()
    assert advisor.model is not None


def test_fit_sets_trained():
    X, y, weights = _data(n_samples=20)
    advisor = CatBoostAdvisor(iterations=10, verbose=False)
    advisor.fit(X, y, weights)
    assert advisor.is_trained is True


def test_fit_no_weights():
    X, y, _ = _data(n_samples=20)
    advisor = CatBoostAdvisor(iterations=10, verbose=False)
    advisor.fit(X, y)
    assert advisor.is_trained is True


def test_fit_float64():
    X, y, _ = _data(n_samples=20)
    advisor = CatBoostAdvisor(iterations=10, verbose=False)
    advisor.fit(X.astype(np.float64), y.astype(np.float64))
    assert advisor.is_trained is True


def test_predict_shape(advisor_fit):
    X, _, _ = _data(n_samples=1)
    result = advisor_fit.predict(X[0])
    assert result.ndim == 1
    assert result.shape == (N_GENES,)


def test_predict_dtype(advisor_fit):
    X, _, _ = _data(n_samples=1)
    result = advisor_fit.predict(X[0])
    assert result.dtype == np.float32


def test_predict_untrained_error():
    advisor = CatBoostAdvisor()
    x = np.zeros(N_FEATS, dtype=np.float32)

    with pytest.raises(RuntimeError, match="not trained"):
        advisor.predict(x)


def test_predict_2d_error(advisor_fit):
    X, _, _ = _data(n_samples=2)

    with pytest.raises(ValueError, match="1-dimensional"):
        advisor_fit.predict(X)


def test_save_untrained_error(tmp_path):
    advisor = CatBoostAdvisor()

    with pytest.raises(RuntimeError, match="Cannot save"):
        advisor.save(tmp_path / "model.cbm")


def test_save_creates_file(advisor_fit, tmp_path):
    path = tmp_path / "model.cbm"
    advisor_fit.save(path)

    assert path.exists()


def test_save_creates_dirs(advisor_fit, tmp_path):
    nested = tmp_path / "a" / "b" / "c" / "model.cbm"
    advisor_fit.save(nested)

    assert nested.exists()


def test_load_sets_trained(advisor_fit, tmp_path):
    path = tmp_path / "model.cbm"
    advisor_fit.save(path)

    fresh = CatBoostAdvisor()

    assert fresh.is_trained is False

    fresh.load(path)

    assert fresh.is_trained is True


def test_load_missing_file_error():
    advisor = CatBoostAdvisor()

    with pytest.raises(FileNotFoundError):
        advisor.load("/nonexistent/path/model.cbm")


def test_save_load_keeps_predictions(advisor_fit, tmp_path):
    path = tmp_path / "model.cbm"
    advisor_fit.save(path)

    loaded = CatBoostAdvisor()
    loaded.load(path)

    X, _, _ = _data(n_samples=1)

    assert advisor_fit.predict(X[0]) == pytest.approx(loaded.predict(X[0]), abs=1e-5)