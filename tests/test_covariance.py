from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from orbitguard.orbit.covariance import parse_stk_ephemeris

TRI = " ".join(["4.0", "0.5", "1.0", "0.0", "0.0", "9.0"] + ["0.0"] * 15)
TRI2 = " ".join(["8.0", "0.0", "2.0", "0.0", "0.0", "9.0"] + ["0.0"] * 15)
STK = f"""stk.v.12.0
BEGIN Ephemeris
# comment
    ScenarioEpoch                 20 Oct 2026 20:28:06.514
    CoordinateSystem              TEMEOfDate
    EphemerisTimePosVel

0.0 7000.0 0.0 0.0 0.0 7.5 0.0
180.0 6990.0 1350.0 0.0 -1.4 7.4 0.0

    CovarianceTimePosVel

0.0 {TRI}
180.0 {TRI2}

End Ephemeris
"""


def test_parse_stk_epoch_states_and_symmetric_covariance():
    e = parse_stk_ephemeris(STK, 1)
    assert e.epoch == datetime(2026, 10, 20, 20, 28, 6, 514000, tzinfo=timezone.utc)
    assert e.rv.shape == (2, 6) and e.cov.shape == (2, 6, 6)
    c = e.cov[0]
    assert np.allclose(c, c.T)
    assert c[0, 0] == 4.0 and c[1, 0] == 0.5 and c[1, 1] == 1.0 and c[2, 2] == 9.0


def test_covariance_interpolates_and_is_bounded_in_time():
    e = parse_stk_ephemeris(STK, 1)
    mid = e.covariance_at(e.epoch + timedelta(seconds=90))
    assert mid[0, 0] == pytest.approx(6.0) and mid[1, 1] == pytest.approx(1.5)
    assert e.covariance_at(e.epoch - timedelta(seconds=1)) is None
    assert e.covariance_at(e.epoch + timedelta(seconds=181)) is None


def test_rejects_file_without_covariance():
    with pytest.raises(ValueError):
        parse_stk_ephemeris("stk.v.12.0\nBEGIN Ephemeris\nEnd Ephemeris\n")
