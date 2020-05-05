# Plugins

In order to switch between executing on Slurm and another cluster,
`submitit` provides a plugin API.
Each plugin must implement an `Executor`, a `Job`, an `InfoWatcher` and a `JobEnvironment` class.
Look at [structure.md](./structure.md) for more details on those classes.

Main functions to implement:
  - `Executor.submit`: from a function create a `Job` using the correct log files and python executable
  - `Executor._convert_parameters`: convert standardized parameters to cluster specific ones
  - `InfoWatcher.get_info`: given a job id, get the state of the job (pending, running, ...)
  - `JobEnviroment`: setup signal handlers and requeuing to behave nicely in the cluster

Look for `@plugin-dev` mention in comments for more details.

Plugins must have an `entry_points.txt` file with the following keys:

```
[submitit]
executor=my_plugin:MyExecutor
job_environment=my_plugin:MyJobEnvironment
```

See [packaging](https://packaging.python.org/guides/creating-and-discovering-plugins/#using-package-metadata) documentation for more details.
