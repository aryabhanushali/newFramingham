"""
Smoke test for the Streamlit demo. Uses streamlit.testing.v1.AppTest, which
runs the script end-to-end with mock widgets and exposes any exception or
st.error block raised during execution.

If this passes, the app is loadable on Streamlit Cloud.
"""

import os

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_streamlit_app_runs_without_errors():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=60).run()

    assert list(at.exception) == [], f"AppTest exceptions: {list(at.exception)}"
    assert list(at.error) == [], f"AppTest error blocks: {list(at.error)}"
