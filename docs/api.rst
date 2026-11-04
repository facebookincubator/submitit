API reference
=============

Executors
---------

.. autoclass:: submitit.AutoExecutor
   :show-inheritance:

.. autoclass:: submitit.Executor
   :members: submit, map_array, batch, update_parameters

.. autoclass:: submitit.SlurmExecutor
   :show-inheritance:

.. autoclass:: submitit.LocalExecutor
   :show-inheritance:

.. autoclass:: submitit.DebugExecutor
   :show-inheritance:

Jobs
----

.. autoclass:: submitit.Job
   :members: job_id, state, done, result, results, exception, wait, cancel, task, num_tasks, stdout, stderr, paths, submission, awaitable

.. autoclass:: submitit.SlurmJob
   :show-inheritance:

.. autoclass:: submitit.core.utils.JobPaths
   :members: folder, submission_file, submitted_pickle, result_pickle, stdout, stderr

.. autoclass:: submitit.JobEnvironment
   :members: job_id, num_tasks, num_nodes, node, global_rank, local_rank, hostname, hostnames, paths

Helpers
-------

.. autoclass:: submitit.helpers.CommandFunction

.. autoclass:: submitit.helpers.DelayedSubmission

.. autoclass:: submitit.helpers.Checkpointable
   :members: __submitit_checkpoint__

.. autoclass:: submitit.helpers.FunctionSequence
   :members: add

.. autofunction:: submitit.helpers.as_completed

.. autofunction:: submitit.helpers.monitor_jobs

.. autofunction:: submitit.helpers.clean_env

.. autoclass:: submitit.helpers.RsyncSnapshot

.. autoclass:: submitit.helpers.TorchDistributedEnvironment
   :members:

Errors
------

.. autoexception:: submitit.core.utils.FailedJobError

.. autoexception:: submitit.core.utils.UncompletedJobError

.. autoexception:: submitit.core.utils.FailedSubmissionError
