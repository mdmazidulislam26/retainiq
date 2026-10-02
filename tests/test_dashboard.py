# Headless smoke test of the Streamlit app (Streamlit AppTest).
from pathlib import Path
from streamlit.testing.v1 import AppTest

DASH = str(Path(__file__).resolve().parents[1] / "src" / "dashboard.py")


# The app renders its worklist, drift table and metrics without errors.
def test_dashboard_runs_without_exception():
    at = AppTest.from_file(DASH, default_timeout=120).run()
    assert not at.exception
    assert len(at.dataframe) >= 2          # worklist + drift table
    assert at.metric[0].value              # metrics rendered
