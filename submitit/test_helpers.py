# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import time
from pathlib import Path

from . import helpers
from .auto.auto import AutoExecutor
from .core import utils


def _three_time(x: int) -> int:
    return 3 * x


def test_function_sequence_checkpoint(tmp_path: Path) -> None:
    file = tmp_path / "test_funcseq.pkl"
    fs0 = helpers.FunctionSequence(verbose=True)
    fs0.add(_three_time, 4)
    fs0.add(_three_time, 5)
    assert len(fs0) == 2
    assert sum(x.done() for x in fs0) == 0
    utils.cloudpickle_dump(fs0, file)
    fs1 = utils.pickle_load(file)
    assert sum(x.done() for x in fs1) == 0
    assert fs1() == [12, 15]
    assert sum(x.done() for x in fs1) == 2


def test_as_completed(tmp_path: Path) -> None:
    ex = AutoExecutor(folder=tmp_path, cluster="local")

    def f(x: float) -> float:
        time.sleep(x)
        return x

    # slow need to be > 1.5s otherwise it might finish before we start polling.
    slow, fast = 1.5, 0.1
    # One slow job and two fast jobs.
    jobs = ex.map_array(f, [slow, fast, fast])
    start = time.time()
    finished_jobs = []
    for n, j in enumerate(helpers.as_completed(jobs, poll_frequency=0.1)):
        elapsed = time.time() - start
        if n < 2:
            # we start getting result before the slow job finished.
            assert elapsed < slow
        finished_jobs.append(j)
    # We get fast job results first, then result of the slow job.
    assert [fast, fast, slow] == [j.result() for j in finished_jobs]
    assert jobs[0] is finished_jobs[-1]
