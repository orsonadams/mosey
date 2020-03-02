Why Mosey?
---

Experiments leave me wanting

* to compare models at a glance
* to run save models that are good
* to track and save experimentation parameters
* to run models async asynchronously 
* to track data used for specific models

So I built mosey an lightweight experimentation pipeline 
that runs on the MOSI dataset but is generalizable to _any_ machine learning
task.

It's of course  experimental!

Key Features
------------

- Name and save experiments
- Track data used in an experiment
-Track parameters as will
- Save model and experimentation statistic
- Run mutiple models  (in sequence for now) but will not bstop on single failures.
- Simple (ish), clean and fast.

Coming
--------

* Serialization for models parameters, datetime objects; the works
* Tests!
* Handle custom data preperation schemes in `Data`
* Compare over experiments, currently compare only over runs
* Load Comparision!
