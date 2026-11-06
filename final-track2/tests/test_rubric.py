"""Exact course weights, independent bonuses, and actual grading-page behavior."""

import math

import pytest

from course_metrics import (
    performance_points,
    POINT_MAXIMA,
    BASE_POINTS,
    REPORT_POINTS,
    REPORT_CRITERION_POINTS,
    RANK_BONUS_POINTS,
    NOVELTY_BONUS_POINTS,
    AUTOMATIC_POINTS,
    AUTOMATIC_MAXIMA,
    RESULTS_EVIDENCE_POINTS,
)


from evaluation.portable_metric import PROTOCOL_SHA256


def scores():
    mean = dict(
        matching=0.75,
        scene_geometry=0.75,
        object_geometry=0.75,
        scene_appearance=0.75,
        object_appearance=0.75,
        collision_free=0.75,
        benchmark=50,
    )
    return dict(
        schema="cs498.track2.split_summary.v2",
        protocol_sha256=PROTOCOL_SHA256,
        means=mean,
        expected=10,
        completed=10,
        missing=[],
        cases=[],
    )


def test_weights_match_final_allocation_and_base_is_twenty():
    assert POINT_MAXIMA == dict(
        results_evidence=8, perception=1.5, reconstruction=1.5, collision_free=1.0
    )
    assert math.fsum(POINT_MAXIMA.values()) == 12
    assert math.fsum(AUTOMATIC_MAXIMA.values()) == AUTOMATIC_POINTS == 4
    assert RESULTS_EVIDENCE_POINTS == 8
    assert REPORT_POINTS == 4 * REPORT_CRITERION_POINTS == 8
    assert math.fsum(POINT_MAXIMA.values()) + REPORT_POINTS == BASE_POINTS == 20
    assert BASE_POINTS + RANK_BONUS_POINTS + NOVELTY_BONUS_POINTS == 24


def test_metric_component_weights_and_equal_splits():
    val, test = scores(), scores()
    reference = dict(
        protocol_sha256=PROTOCOL_SHA256,
        splits=dict(
            val=dict(means=dict(benchmark=50)), test=dict(means=dict(benchmark=50))
        ),
    )
    full = performance_points(val, test, reference)
    assert full == pytest.approx(AUTOMATIC_MAXIMA)
    for summary in [val, test]:
        summary["means"]["object_appearance"] = 0
    assert full["reconstruction"] - performance_points(val, test, reference)[
        "reconstruction"
    ] == pytest.approx(3 / 8)
    test["means"]["matching"] = 0
    test["means"]["collision_free"] = 0
    assert performance_points(val, test, reference)["perception"] == pytest.approx(0.75)
    assert performance_points(val, test, reference)["collision_free"] == pytest.approx(
        0.5
    )


def test_evidence_credit_is_never_automatically_tied_to_reference_score():
    val, test = scores(), scores()
    reference = dict(
        protocol_sha256=PROTOCOL_SHA256,
        splits=dict(
            val=dict(means=dict(benchmark=99)), test=dict(means=dict(benchmark=99))
        ),
    )
    before = performance_points(val, test, reference)
    val["means"]["benchmark"] = test["means"]["benchmark"] = 0
    assert performance_points(val, test, reference) == before
    assert "baseline" not in before and "results_evidence" not in before
