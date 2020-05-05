# Tips and caveats

 - It is always preferable to submit functions defined in a module, the traceback will be more explicit in case of error during the execution.
 - Modules imported after added their paths through `sys.path.append` cannot be correctly pickled. If you can, restructure your code to avoid appending paths this way.
   If you cannot, then an ugly hack consists in using `sys.path.append` lazily, i.e. *within* a function, and then importe the required module.
 - Imports order may not be respected. If this causes issue, you can import lazily as above. Another option is to wrap your import into another module in which the order can be respected.
 - On SLURM, use the flush option of print to avoid logs to be buffered `print(text, flush=True)`.
 - Contributors are much welcome! You'll probably find some weird behaviors, you can open an issue if so, and solve it if you can ;)
 - the API may still evolve, in particular regarding the locations of pickled data and logs and how they are managed. It is preferable to use a fixed version if you do not want to have any compatibility issue at some point. Use the lastest "release" version that suits you.
 - since the pickled function are references to the module function, if the module changes between the submission and the start
 of the computation, then the computation may not be the one you expect. Similarly, if at the start of the computation, one file cannot
 be run (if you are currently editing it for instance), then the computation will fail. Joblib implements some kind of version check it seems,
 which could be handy.
 - Do not hesitate to create your own `Executor` class: this can be useful to control more precisely how your job are submitted.
 In about 10 lines of code, you can for instance have an executor which creates a new logging folder for each submitted jobs etc...
 - Some non-picklable objects like locks cannot be submited. This may cause issue if they are used as default arguments of a function.

## Specific to Slurm
 - While all jobs are requeued after preemption, only Checkpointable classes are requeued after a timeout (since a stateless function is expected to timeout again if it is requeued)
 - Timeouts are requeued a limited number of time (default: 3, see `Executor`) in order to avoid endless jobs.
 - the log/output folder must be chosen carefully: it must be a directory shared between instances, which rules out /tmp. If you
 are not careful with this, you may get job that fails without any log to debug them. Also, it will fill up pretty fast with batch
 files and pickled objects, but no cleaning mechanism is currently implemented.

## Debugging

If you want to add breakpoints to a function run by `submitit`, the easiest way is to `AutoExecutor(cluster="debug")`.
This will execute all the "submitted" jobs inside the main process, and your breakpoints will be hit normally.
