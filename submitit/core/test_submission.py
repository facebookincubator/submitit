import pickle
import weakref
from pathlib import Path

from .submission import save_error, save_result


def test_save_result(tmp_path: Path) -> None:
    result = test_save_result.__name__
    save_result(result, tmp_path / "res.pkl")
    loaded = pickle.loads((tmp_path / "res.pkl").read_bytes())
    assert loaded == ("success", result)


def test_unpickable_result(tmp_path: Path) -> None:
    result = weakref.ref(test_unpickable_result)
    save_result(result, tmp_path / "res.pkl")

    loaded = pickle.loads((tmp_path / "res.pkl").read_bytes())
    assert loaded[0] == "error"
    assert isinstance(loaded[1], TypeError)


def test_save_error(tmp_path: Path) -> None:
    class MyException(Exception):
        pass

    try:
        raise MyException("oopsy")
    except Exception as e:
        save_error(e, tmp_path / "res.pkl")
    loaded = pickle.loads((tmp_path / "res.pkl").read_bytes())
    assert loaded[0] == "error"
    assert isinstance(loaded[1], MyException)
    assert loaded[1].args == ("oopsy",)


def test_unpickable_error(tmp_path: Path) -> None:
    class MyException(Exception):
        pass

    try:
        raise MyException("oopsy", weakref.ref(test_unpickable_error))
    except Exception as e:
        save_error(e, tmp_path / "res.pkl")
    status, trace = pickle.loads((tmp_path / "res.pkl").read_bytes())
    assert status == "error"
    assert isinstance(trace, str)
    assert "Traceback" in trace
    assert "oopsy" in trace
    assert "weakref" in trace
