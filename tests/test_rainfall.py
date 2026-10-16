import numpy as np
import pytest
from scipy.spatial.distance import cosine

from src.rainfall import (
    CropRule, Region, RAINFALL_DATA, cosine_similarity, default_crop_rules,
    detect_rainy_seasons, euclidean, month_labels, pairwise_matrix, pearson,
    rainfall_regime,
)


def _regions():
    return [Region(n, v) for n, v in RAINFALL_DATA.items()]


def test_region_basics():
    r = Region("X", [10] * 11 + [130])
    assert r.annual_total() == 240
    assert r.wettest_month() == "Dec" and r.driest_month() == "Jan"
    assert r.cv() > 0


@pytest.mark.parametrize("bad", [[1] * 11, [1] * 13, [-1] + [1] * 11, [float("nan")] + [1] * 11])
def test_region_rejects_bad_data(bad):
    with pytest.raises(ValueError):
        Region("bad", bad)


def test_zero_rainfall_cv_raises():
    with pytest.raises(ValueError):
        Region("dry", [0] * 12).cv()


def test_crop_rule_classification_and_validation():
    rule = CropRule("maize", 125, 200)
    assert rule.classify(50) == "Drought risk"
    assert rule.classify(125) == "Good" and rule.classify(200) == "Good"
    assert rule.classify(300) == "Waterlogging risk"
    with pytest.raises(ValueError):
        CropRule("bad", 200, 100)
    with pytest.raises(ValueError):
        rule.classify(-1)


def test_default_rules_and_labels_cover_12_months():
    rules = default_crop_rules()
    assert {r.crop for r in rules} >= {"maize", "beans", "coffee"}
    assert all(r.source for r in rules)
    assert len(month_labels(_regions()[0], rules)) == 12


def test_cosine_matches_scipy():
    a, b = np.array([1.0, 2, 3]), np.array([4.0, 0, 1])
    assert cosine_similarity(a, b) == pytest.approx(1 - cosine(a, b))
    for r1, r2 in [(x.rainfall, y.rainfall) for x in _regions() for y in _regions()]:
        assert cosine_similarity(r1, r2) == pytest.approx(1 - cosine(r1, r2))


def test_cosine_ignores_magnitude_but_euclidean_does_not():
    a = np.array([1.0, 2, 3])
    assert cosine_similarity(a, 10 * a) == pytest.approx(1.0)
    assert pearson(a, 10 * a) == pytest.approx(1.0)
    assert euclidean(a, 10 * a) > 30


def test_cosine_zero_vector_and_shape_errors():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 2])
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_pairwise_matrix_symmetric_unit_diagonal():
    m = pairwise_matrix(_regions(), cosine_similarity)
    assert np.allclose(m, m.T) and np.allclose(np.diag(m), 1)


def test_peak_detection_known_shapes():
    uni = Region("uni", [0, 0, 0, 0, 10, 50, 100, 50, 10, 0, 0, 0])
    bi = Region("bi", [0, 0, 100, 20, 0, 0, 0, 20, 100, 0, 0, 0])
    assert rainfall_regime(uni) == "unimodal"
    assert rainfall_regime(bi) == "bimodal"
    assert detect_rainy_seasons(bi) == ["Mar", "Sep"]


def test_season_straddling_year_end_is_one_peak():
    r = Region("wrap", [100, 40, 0, 0, 0, 0, 0, 0, 0, 0, 40, 80])
    assert rainfall_regime(r) == "unimodal"
