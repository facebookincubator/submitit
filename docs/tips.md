# Tips and caveats

 - Imports order may not be respected. If this causes issue, you can import lazily, i.e. *within* the submitted function. Another option is to wrap your import into another module in which the order can be respected.
 - Contributors are much welcome! You'll probably find some weird behaviors, you can open an issue if so, and solve it if you can ;)
 - the API may still evolve, in particular regarding the locations of pickled data and logs and how they are managed. It is preferable to use a fixed version if you do not want to have any compatibility issue at some point. Use the lastest "release" version that suits you.
 - since the pickled function are references to the module function, if the module changes between the submission and the start
 of the computation, then the computation may not be the one you expect. Similarly, if at the start of the computation, one file cannot
 be run (if you are currently editing it for instance), then the computation will fail. Joblib implements some kind of version check it seems,
 which could be handy.
 - Do not hesitate to create your own `Executor` class: this can be useful to control more precisely how your job are submitted.
 In about 10 lines of code, you can for instance have an executor which creates a new logging folder for each submitted jobs etc...

## Specific to Slurm

 - While all jobs are requeued after preemption, only Checkpointable classes are requeued after a timeout (since a stateless function is expected to timeout again if it is requeued)
