
Experimention framework
---------------------

research
    - [google machine learning](https://developers.google.com/machine-learning/guides/rules-of-ml/)
    - 
    - existing frameworks
        - mlflow
        - michealangelo UBER
        - flyte, LYft
        - metaflow, netflix
        - 



Wish List
----------


    - specific to the MOSI dataset
        - multi-modal
            - how to handle audio, video and text features holistically
    - be able to split data
    - be able to choose metrics [ global metrics ] 
    - be able to choose model ( use multiple models)
    - be able to use multiple types of models
        - how to combine models of different tasks types
    - do able to compare models
    - train models 
    - system agnostic, language agnostic ( ? )
    - easily extensible, easy to use.
    - distributed training
    -   easy to run on single / multiple machines
    - should easily save model/load model easily
    - save load metrics
    - Should be reliable ( tests, docs )
    - should be lightweight
        - this is a bit ambiguous ( should speak with team about what is lightweight)
            - is sklearn lightweight?
    - python support ( should it support py < 3.5 )
    - Visualization support
    - Support evalution on custom model. 
        - say a team member rights a non sklearn model ( keras, pytorch ... ) they should be able to evaluate on our dataset.
    - <s> Support new test data </s>
    
Hard Requirements
-----------------

    - Should train models of differnt types
    - Should automate
        - train / test spliting
        - training / 
        - evaluation  ( ease of use)
        - black box this but support inspection, modular but not overly so
        - be as declarative as possibe. model.fit(**kwargs)
    - Should be setup for extensibilty ( tests, Object oriented ), docs!
    - Basic, visualisation support for model comparison ( tables mostly)
    - Should fail fast but verbosely by default.
    
    
    
Contraints
-----------------
5 hours. Python shop,



Notes
------------

### background
-------------

Had recently given a talk on mlflow, Had a sense of what's working for core needs


Given the timeframe, should re-use something that exists if possible. 
Use something that team is familiar with (if it breaks, engineers can dig)
Questions are easily answered on stackoverflow. 

For this project I choose to use the sklearn package as the core building block 
It satifies my main concerns and contraints. 
    - Can ship in timeframe
    - Extensible - team is super familiar, lightweight(ish) would rewrite pieces
    - Provides alot of what we need out of the box
        - reliable, APIs for model selection, comparison, loading and saving model,
    -d

Build something that I can use to do assignment!

- Need feature union for different input modes - Encapsulate this, since we're only using the MOSI dataset, it's okay to use something like `data.union()` but  allow user to write another loader and implement `union()`
- user should be able to specify metrics ( use sklearn's metrics types as baseline)
- allow user to provide own metrics ( registration on model class would be nice but for now allow, user to provide custom metrics as functional parameters.
- Seperate data and model encapsulation (pytorch / keras )

### API:
-----------
I do realise this is effectively auto-sklearn
Things we're building top of sklearn, 
	easy feature_union
	visualiztion for model comparison
	a data iterator
	extension for adding models to current build 
		(for comparison)
	a nice API for ensembles and comparison
To train a mosey model.
Be opinionated about the naming conventions becuase things can get ugly fast (spacy style)

```python
# data_iterator is an iterator of lines from the various modes
# provide train, test

dataiterator = DataIterator(models)
search = Search(type=(), cv, jobs=-1)
# set up for models training model(s)
model = Model(dataiterator, 
              models=[],  # 
              metrics=[], # list of metrics for current build
              checkpoint = # save model at regular intervals 
              search=search) 
model.predict(X, batch_size=10)
model.evalute(X, y, show_metrics=True) # show all metrics
model.models['svr'] # show svr specific metrics
model.best(metric) # show best model metrics
model.save(name, filename) # save model with name `name` to `filename`
# load model
model =  model.load(filename)

# extend to additional tasks
# trains a new model of task type `type` over `data` and
# allows comparision with previously trained models
model.extend(type='regression|classification', data=iterator)
