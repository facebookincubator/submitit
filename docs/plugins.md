# Plugins

To support a scheduler other than Slurm, a plugin package implements subclasses of `Executor`, `Job`, `InfoWatcher` and `JobEnvironment` (search the code for `@plugin-dev` comments):

- `Executor`: a `FooExecutor` is selected with `cluster="foo"`, and its parameters are prefixed by `foo_`.
- `Executor.affinity()`: score used by `AutoExecutor` to pick the executor, the highest wins and negative means unavailable.
- `Executor.submit`: creates a `Job` from a function, using the correct log files and Python executable.
- `Executor._convert_parameters`: converts generic parameters (e.g. `timeout_min`) to the cluster's own, renaming them through `_equivalence_dict` by default.
- `Job`: accesses the state, logs and result of a job, using its `InfoWatcher`.
- `InfoWatcher.get_info`: returns the state of a job from its id (pending, running...).
- `JobEnvironment`: maps the cluster's environment variables to the job id and ranks through `_env`, tells with `activated()` whether the job runs on this cluster, and sets up signal handlers and requeuing.

The plugin registers them as entry points in its `pyproject.toml`, see the [packaging guide](https://packaging.python.org/guides/creating-and-discovering-plugins/#using-package-metadata):

```toml
[project.entry-points.submitit]
executor = "my_plugin:MyExecutor"
job_environment = "my_plugin:MyJobEnvironment"
```
