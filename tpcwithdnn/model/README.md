# DNN model classes and functions


## Helper functions

There are a few common model related helper functions which are explained in the following.

### Constructing a model

A model can be constructed with

```python
construct_model(constructor, *args, **kwargs)
```

The first argument `constructor` has to be callable and `args` and `kwargs` are forwarded to that. If there are no `kwargs` and `args` is of length `1`, it is assumed that this is a configuration dictionary like the following:

```python
{"model_args": ((self.grid_phi, self.grid_r, self.grid_z, self.dim_input),),
 "model_kwargs": {"depth": self.depth,
                  "bathnorm": self.batch_normalization,
                  "pool_type": self.pooling,
                  "start_ch": self.filters,
                  "dropout": self.dropout},
 "compile": {"loss": self.lossfun,
             "optimizer": Adam,
             "optimizer_kwargs": {"lr": self.adamlr},
             "metrics": [self.metrics]},
 "fit": {"workers": 1,
         "use_multiprocessing": True,
         "epochs": self.epochs}}
```

where the keys `model_args` and `model_kwargs` are digested instead.



## Bayesian optimisation

The class `KerasBayesianOpt` provides all the functionality to do an Bayesian optimisation. It can be utilised as it's done in `../dnnoptimiser.py`:

```python

bayes_opt = KerasBayesianOpt(self.make_model_config(), self.make_opt_space())
bayes_opt.train_gen = fluctuationDataGenerator(self.indexmatrix_ev_mean_train,
                                               **self.params)
bayes_opt.val_gen = fluctuationDataGenerator(self.indexmatrix_ev_mean_test,
                                             **self.params)
bayes_opt.construct_modeli_func = construct_model
bayes_opt.model_constructor = UNet

bayes_opt.optimise()

out_dir = "./optimisation_output"
if not os.path.exits(out_dir):
    os.makedirs(out_dir)
bayes_opt.save(out_dir)
bayes_opt.plot(out_dir)
```
`self.make_model_config()` returns a configuration dictionary just like the above and `self.make_opt_space()` returns the `hyperopt` space used to draw parameters from and is currently just implemented in `../dnnoptimiser.py` as

```python

@staticmethod
def make_opt_space():
    return {"compile":
            {"optimizer_kwargs": 
                {"lr": hp.uniform("m_learning_rate", 0.0005, 0.002)}}}
```
