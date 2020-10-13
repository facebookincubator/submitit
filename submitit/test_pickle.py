from weakref import ref

import pytest

from .local.debug import DebugExecutor
from .local.local import LocalExecutor


def job_with_weakref(ex):
    class MyObject:
        hello = "world"

    a = MyObject()
    assert ref(a)().hello == "world"

    def f(a_ref):
        a = a_ref()
        assert a is not None
        return a_ref().hello

    return ex.submit(f, ref(a))


@pytest.mark.xfail(reason="'a' is GC-ed before we call the function")
def test_weakref_no_pickle(tmp_path):
    ex = DebugExecutor(tmp_path)
    assert job_with_weakref(ex).result() == "world"


@pytest.mark.xfail(reason="'ref(a)' can't be pickled")
def test_weakref_with_pickle(tmp_path):
    ex = LocalExecutor(tmp_path)
    assert job_with_weakref(ex).result() == "world"
