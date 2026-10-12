import hashlib
from unittest import mock

import requests

import src.data.gdc as gdc

DATA = b"x" * 3000


class _Resp:
    def __init__(self, start: int, fail: bool):
        self.start, self.fail = start, fail
        self.status_code = 206 if start else 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk):
        yield DATA[self.start : self.start + 1000]
        if self.fail:
            raise requests.exceptions.ChunkedEncodingError("IncompleteRead")
        yield DATA[self.start + 1000 :]


def test_download_resumes_after_dropped_connection(tmp_path):
    offsets = []

    def fake_get(url, headers, stream, timeout):
        start = int(headers["Range"][6:-1]) if headers else 0
        offsets.append(start)
        return _Resp(start, fail=len(offsets) < 3)

    with mock.patch.object(gdc.requests, "get", fake_get), mock.patch.object(gdc.time, "sleep"):
        out = gdc.download_file("id", hashlib.md5(DATA).hexdigest(), tmp_path / "f.svs")
    assert offsets == [0, 1000, 2000]
    assert out.read_bytes() == DATA
    assert not (tmp_path / "f.svs.part").exists()
