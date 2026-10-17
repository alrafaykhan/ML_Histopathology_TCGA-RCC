import pandas as pd

from src.data.idc import verify_series
from src.utils import path_bytes, wsi_open_path


def test_verify_series_and_dicom_paths(tmp_path):
    series = tmp_path / "1.2.3"
    series.mkdir()
    (series / "b.dcm").write_bytes(b"x" * 10)
    (series / "a.dcm").write_bytes(b"x" * 5)
    expected = pd.DataFrame({"crdc_instance_uuid": ["a", "b"], "instance_size": [5, 10]})
    assert verify_series(series, expected) == ""
    assert "wrong size" in verify_series(series, expected.assign(instance_size=[5, 11]))
    extra = pd.concat([expected, pd.DataFrame({"crdc_instance_uuid": ["c"], "instance_size": [1]})])
    assert "missing" in verify_series(series, extra)
    assert wsi_open_path(series).name == "a.dcm"
    assert path_bytes(series) == 15 and path_bytes(tmp_path / "absent") == 0
