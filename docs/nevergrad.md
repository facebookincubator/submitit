# Hyperparameter search with nevergrad

[nevergrad](https://github.com/facebookresearch/nevergrad) is a derivative-free optimization library, e.g. for hyperparameter search.
Its optimizers accept a submitit executor to evaluate several points in parallel, each as a cluster job:

```python
import nevergrad as ng
import submitit


def train(learning_rate: float, dropout: float) -> float: ...  # train a model and return its validation loss


parametrization = ng.p.Instrumentation(
    learning_rate=ng.p.Log(lower=1e-4, upper=1.0),
    dropout=ng.p.Scalar(lower=0, upper=1),
)
optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=128, num_workers=16)
executor = submitit.AutoExecutor(folder="logs/%j")
executor.update_parameters(timeout_min=60, gpus_per_node=1, slurm_partition="<your_partition>")
recommendation = optimizer.minimize(train, executor=executor)  # at most 16 jobs at a time
print(recommendation.kwargs)
```

Instead of `minimize`, to control the loop (with `train`, `parametrization` and `executor` from above):

```python
optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=128, num_workers=16)
for _ in range(optimizer.budget // optimizer.num_workers):
    candidates = [optimizer.ask() for _ in range(optimizer.num_workers)]
    jobs = [executor.submit(train, *c.args, **c.kwargs) for c in candidates]
    for candidate, job in zip(candidates, jobs):
        optimizer.tell(candidate, job.result())
print(optimizer.provide_recommendation().kwargs)
```

Each evaluation is a cluster job: use this for evaluations lasting tens of minutes or more.
See [nevergrad's documentation](https://facebookresearch.github.io/nevergrad/) for its optimizers and parametrization.
