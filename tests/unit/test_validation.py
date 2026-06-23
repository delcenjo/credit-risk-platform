import pytest

from credit_scoring.data.validation import DataQualityError, validate_frame


def test_clean_frame_passes(make_frame):
    report = validate_frame(make_frame())
    assert report.ok
    assert report.rows > 0
    assert 0 <= report.default_rate <= 1


def test_missing_column_is_rejected(make_frame):
    frame = make_frame().drop(columns=["age"])
    with pytest.raises(DataQualityError):
        validate_frame(frame)


def test_out_of_range_repayment_status_is_flagged(make_frame):
    frame = make_frame()
    frame.loc[0, "pay_0"] = 50
    report = validate_frame(frame, raise_on_error=False)
    assert not report.ok
    assert any("pay_0" in issue for issue in report.issues)
