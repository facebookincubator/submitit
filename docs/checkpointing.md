# Checkpointing


## The basics of checkpointing with `submitit`

Checkpointing is trickier and requires a precise understanding of the inner working of the job pickling.

At the time we need to requeue a job (after preemption or timeout), we can edit the submitted task according to the current state of the computation. Since in a standard function, the state cannot be accessed, we need to submit a callable (an instance of a class with a `__call__` method) instead of a mere function.

In practice, when requeuing, `submitit` will check if the callable has a `__submitit_checkpoint__` or `checkpoint` method.
If so, it will send it the initial arguments and the `checkpoint` method takes care of preparing the new submission.
The `checkpoint` method must therefore have a signature able to receive all parameters from the `__call__` function of your callable.
It must return a `DelayedSubmission` which acts exactly as `executor.submit`: you can provide any function and arguments. Alternatively, it can return `None` if for some reason it does not want to be requeued.

## Minimal example

Typically, in most cases you would just resubmit the current callable at its current state with the same initial arguments, so adding the
following generic `checkpoint` method to your callable may work just fine:
```python
def checkpoint(self, *args: Any, **kwargs: Any) -> submitit.helpers.DelayedSubmission:
    return submitit.helpers.DelayedSubmission(self, *args, **kwargs)  # submits to requeuing
```
If this kind of checkpoint is sufficient for you, you can derive your callable from `submitit.helpers.Checkpointable` which implements this very function.

Generally checkpointing requires a modification of your code to skip the parts that have been done before being rescheduled.
You can look at [the MNIST example](./mnist.py) to see what it looks like in practice.

You may however submit something completely different if you wish. This can happen for instance if:
 - you want to restart with different parameters.
 - you do not want all the attributes to be pickled. Typically, you may want to dump a neural network in a separate file
  in a standard format and set the corresponding argument to None in order to avoid relying on pickle for saving the
  network.

For a basic example of a checkpointable callable, checkout the code from `submitit.helpers.FunctionSequence` in [helpers.py](../submitit/helpers.py).


## Example - Training and checkpointing a model

The following example provides a recipe for checkpointing the training of a model. It is more complex because we do not want to rely on `submitit` to pickle the model. This recipe has not been fool proofed yet, I am happy to help if you encounter any issue ;)

```python
from pathlib import Path
import submitit

class NetworkTraining:

    def __init__(self):
        # this is the "state" which we will be able to access when checkpointing:
        self.model = None

    def __call__(self, checkpointpath: str):
        if not Path(checkpointpath).exists():
            self.model = ... # initialize your model
        else:
            self.model = ... # load your model
        # train your model
        ...

    def checkpoint(self, checkpointpath: str) -> submitit.helpers.DelayedSubmission:
        # the checkpoint method is called asynchroneously when the slurm manager
        # sends a preemption signal, with the same arguments as the __call__ method
        # "self" is your callable, at its current state.
        # "self" therefore holds the current version of the model:
        # do whatever you need to do to dump it properly
        self.model.dump(checkpointpath)  # this is an example that probably does not work
        ...
        # create a new, clean (= no loaded model) NetworkTraining instance which
        # will be loaded when the job resumes, and will fetch the dumped model
        # (creating a new instance is not necessary but can avoid weird interactions
        # with the current instance)
        training_callable = NetworkTraining()
        # Resubmission to the queue is performed through the DelayedSubmission object
        return submitit.helpers.DelayedSubmission(training_callable, checkpointpath)
```

When you want to train your model, you just have to run the following code, and it will be
submitted to a slurm job, which will be checkpointed and requeued at most `slurm_max_num_timeout=3` times if timed out
(and any number of time if preempted):
```python
import submitit
from .network import NetworkTraining  # must be defined in an importable module!
executor = submitit.AutoExecutor(folder="logs_training", slurm_max_num_timeout=3)
executor.update_parameters(timeout_min=30, slurm_partition="your_partition",
                           gpus_per_node=1, cpus_per_task=2)
training_callable = NetworkTraining()
job = executor.submit(training_callable, "some/path/for/checkpointing/your/network")
```

On Slurm cluster, you can trigger a fake preemption or timeout in order to test your checkpointing by using the job method: `job._interrupt(timeout=<False,True>)`.
