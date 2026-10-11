import pytest


def test_preview_mailbox_keeps_only_latest_frame():
    pytest.importorskip("PySide6")
    from liveswap.client.gui import Runner
    runner = Runner(None, None)
    for index in range(1000):
        runner.cache_preview(index)
        runner.cache_status({"frame": index})
    status, preview = runner.poll()
    assert status == {"frame": 999}
    assert preview == 999
    assert runner.poll() == (None, None)
