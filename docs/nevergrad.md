# Using `nevergrad` with `submitit`

`nevergrad` is a derivative-free optimization toolbox developed at FAIR which can be used to tune network hyperparameters.
These algorithms can be competitive over random search if you have around 10 parameters or more.

## Basics of `nevergrad`

The following is a simplified version of the tutorial in [`nevergrad`'s repository](https://github.com/facebookresearch/nevergrad/README.md), you can find more details there.

### Example of optimization

For the sake of this example, we'll define a function to optimize:
```python
def myfunction(x, y=12):
    return sum((x - .5)**2) + abs(y)
```

Before we can perform optimization, we must define how this function is instrumented, i.e. the values that the parameters
of the function can take. This is done through the `Instrumentation` class.
The following states the first parameter of the function is an array of size 2, and the second a float:

```python
import nevergrad as ng
instrum = ng.p.Instrumentation(
    ng.p.Array(shape=(2,)),  # first parameter
    y=ng.p.Scalar())  # second (named) parameter
```


Then you can initialize an algorithm (here `TwoPointsDE`, a full list can be obtained with `list(ng.optimizers.registry.keys())` with
this instrumentation and the budget (number of iterations) it can spend, and run the optimization:
```python
import nevergrad as ng
optimizer = ng.optimizers.TwoPointsDE(parametrization=instrum, budget=100)
recommendation = optimizer.minimize(square)
print(recommendation.value)
>>> (array([0.500, 0.499]),), {y: -0.012}
```
`recommendation` holds the optimal attributes `args` and `kwargs` found by the optimizer for the provided function.
The optimal value is obtained through `recommendation.value`.

### Instrumentation


5 base types of variables are currently provided for instrumentation:
- `Choice`: for unordered categorical values.
- `TransitionChoice`: for ordered categorical values.
- `Array`: for array parameters, possibly bounded
- `Scalar`: for standard scalar parameters
- `Log`: for log-distributed scalar parameters

Here is a basic example:
```python
import nevergrad as ng

arg1 = ng.p.TransitionChoice(["a", "b"])  # either "a" or "b" (ordered)
arg2 = ng.p.Choice(["a", "c", "e"])  # "a", "c" or "e" (unordered)
arg4 = ng.p.Array(shape=(4, 3)).set_bounds(lower=0, upper=1)   # an array of size (4, 3) with values between 0 and 1

# the following instrumentation uses these variables (and keeps the 3rd argument constant)
instrum = ng.p.Instrumentation(arg1, arg2, "constant_argument", arg4)
```

And this is a more realistic instrumentation example for a neural network training:
```python
instru = ng.p.Instrumentation(
    dataset=ng.p.Choice(['wikilarge', 'allnli']),
    # Arch
    architecture=ng.p.Choice(['fconv', 'lightconv', 'fconv_self_att', 'lstm', 'transformer']),
    dropout=ng.p.Scalar(lower=0, upper=1)
    # Training
    max_tokens=ng.p.TransitionChoice(np.arange(500, 20001, 100).tolist()),
    max_epoch=ng.p.Scalar(lower=1, upper=50).set_integer_casting(),
    # Optimization
    lr=ng.p.Log(lower=0.001, upper=1.0),
)
```


## Working asynchronously with submitit

To speed up optimization, you probably want to run several function evaluations concurrently.
To do this, you need to notify `nevergrad` at the optimizer initialization that you will have several workers (example: 32 here):

```python
optimizer = ng.optimizers.TwoPointsDE(parametrization=instru, budget=8192, num_workers=32)
```
This, way, the optimizer will be prepared to provide several points to evaluate at once. You have then 2 ways to handle these evaluations:


### Through the optimize method

The `minimize` method takes an executor-like object which is compatible with `submitit` (and `concurrent.futures` and `dask`).
With the following, nevergrad will take care of submitting a job per function evaluation, with at most 32 jobs in parallel:
```python
executor = AutoExecutor(folder=my_folder)
executor.update_parameters(timeout_min=60, gpus_per_node=1, cpus_per_task=2)
recommendation = optimizer.minimize(my_function, executor=executor, verbosity=2)
```

### Using the ask and tell interface

`nevergrad` also provides an ask and tell interface:
- `ask()` provides a candidate point to evaluate.
- `tell(x, value)` is used to feed the function evaluation back to `nevergrad`.

In this case you are the one responsible for properly running the optimization procedure. A naive loop with batch evaluation could like like this:
```python
remaining = optimizer.budget - optimizer.num_ask
while remaining:
    candidates = [optimizer.ask() for k in range(remaining)]
    jobs = [executor.submit(my_function, *c.args, **c.kwargs)) for c in candidates]
    for candidate, job in zip(candidates, jobs):
        optimizer.tell(candidate, job.result())
    remaining = optimizer.budget - optimizer.num_ask
recommendation = optimizer.provide_recommendation()
```

### Gotcha

Since a job will be submitted for each function evaluation, using `submitit` executor in `nevergrad` is suitable for evaluations which take at least tens of minutes, not for small evaluations which will overload the cluster and spend more time pending than running.
