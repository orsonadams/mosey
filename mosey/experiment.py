import logging as logger
import getpass
from contextlib import contextmanager
import datetime
import os
import json
import joblib
from pandas import DataFrame
import uuid

# TODO logging to file as well.
logger.basicConfig(level=logger.DEBUG)


# TODO extend to handling spliting in the data class with arbitrary strategy
class Data:
    def __init__(self, 
                 train=None, 
                 test=None, 
                 val=None, 
                 target=None, 
                 random_state=147):

        # NOTE extend to handle slicing out specific feature columns
        self.random_state = random_state
        self.target = target
        self.xtrain, self.ytrain = self._get_X_y(train)
        if test is not None:
            self.xtest, self.ytest = self._get_X_y(test)
        else:
            self.xtest, self.ytest = None, None
        if val is not None:
            self.xval, self.yval = self._get_X_y(val)
        else:
            self.xval, yval = None, None
        # train data must be provided

    def _check(self, X, y):
        raise NotImplementedError 
    
    def _get_X_y(self, data):
        if not self.target:
            logger.info('No target provided, gatheing features')
        else:
            if hasattr(data, self.target):
                return data, data[self.target] 
            else:
                raise TypeError((f'Target {self.target} not found in'
                                 ' f{data.__name__}'))



class Experiment:
    DEFAULT_BINARY_METRICS = ['precision_score', 'f1_score', 'recall_score']
    def __init__(self, 
                 data=None,
                 task_type=None,
                 metrics=None,
                 experiment_name=None,
                 save_file=None,
                 random_state=None,
                 search=None,
                ):
        
        self.data = data
        self.experiment_name = experiment_name or None 
        self.filename = save_file # persist experiment results
        self.metrics = metrics
        self.statistics = {}
        self.runs = {}
        self.experiment_status = None
        self.computed_scores = None
        self.task_type = task_type
  
        # may not need / want to do hyperamater search
        # TODO broadcast parameter search across machines
        if search and callable(search):
            self.search = search
        else:
            logger.info('No hyperparameter search will be performed')

    @staticmethod
    def _convert_binary_to_multiclass(func, true, preds, average=None):
        return func(true, preds, average=average)     
    
    def _get_requirements(self):
        return 

    def _generate_experiment_id(self):
        return uuid.uuid4().hex 
   
    def __enter__(self):

        self.experiment_id = self._generate_experiment_id()
        self.user = getpass.getuser() 
        self.experiment_start = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
        return self


   # NOTE if an exception is thrown in an `Experiment` 
   #      that is, if  a top level experiment throws a  error
   #      the process will be  killed. Currenttly in a particularly unfriendly
   #      manner
    def __exit__(self, type, value,  traceback):
        
       if type:
           self.experiment_status = 'FAIL'
           logger.error(traceback)
           raise type("Error occured while running the the experiment") 
       else:
           self.experiment_status = 'SUCCESS'
       return self

    @contextmanager
    def run(self, name, **kwargs):
        # TODO allow job based runs  
        #      Figure out how to better handle run exceptions
        
        _run = {}
        _run['name'] = name
        _run['id'] =  self._generate_experiment_id()
        _start = datetime.datetime.now()
        _run['start_time'] = _start.strftime('%Y-%m-%d-%H:%M')
        try:
            yield _run
            runtime = datetime.datetime.now() - _start
            minutes = runtime.seconds // 3600,  
            _run['runtime'] = f'Minutes: {minutes}, Seconds: {runtime.seconds}' # return minutes
            _run['message'] = f' {name} successful. Run delta: {runtime}'
            self.runs[name] = _run
        except Exception as ex:
            msg = f'Error {ex} at run {name}'
            logger.error(msg)
            _run['message'] = msg
            self.runs[name] = _run
        return True       

    # TODO Don't recompute all metrics if they've been computed already
    def compare(self, metrics=None, data=None):
        if metrics:
            self.metrics += metrics
        if not self.metrics and not metrics:
            logger.warn('No metrics avaiable for compute')
        
        if not self.runs:
            logger.info('No runs available for compute')
        return self._compare() 


    def draw(self):
       
        if not self.computed_scores:
            raise ValueError('No scores to compare')
        heading = ['run_name'] + [x.__name__ for x in self.metrics]
        table  = []
        # heading = scores[0].values().values()
        for score in self.computed_scores:
            row = []
            for name, scores in score.items(): # run name, score
                row.append(name)
                for k, v in scores.items():
                    row.append(v)
            table.append(row)
        return DataFrame(table, columns=heading) 
   
    def _compare(self):
        scores = []
        for name, run in self.runs.items():
            run_scores = {}
            try:
                if not run.get('model'):
                    logger.info(f'Run {name} did not cache model')
                    continue
                
                # if hasattr(run['model'], 'predict_proba'):
                #     _predict = run['model'].predict_proba
                elif hasattr(run['model'], 'predict') :
                    _predict = run['model'].predict

                # compute scores for possible metrics in metrics
                # TODO handle mismatch between metric and task
                x, y  = self.data.xval, self.data.yval.values
                preds = _predict(x)
                for m in self.metrics:
                    if m.__name__ in Experiment.DEFAULT_BINARY_METRICS:
                        run_scores[m.__name__] = \
                            Experiment._convert_binary_to_multiclass(m, y,
                                                                     preds,
                                                                     average='micro')
                    else:
                        run_scores[m.__name__] = m(y, preds)
                scores.append({name : run_scores})
            except Exception as ex:
                logger.error(ex)
                logger.error((f'Failed to generate statistics for run {name} '))
        self.computed_scores = scores
        return self.draw()
                
    
    def save(self, filename=None):
        """serialize experiment object if possible else suggest persisting
           experiment_statistics
           Attempts to save  the best model by default using sklearn.joblib
        """
        try:
            current_working_directory = os.path.dirname(os.path.realpath(__file__))
            name = self.experiment_name.replace(' ', '_').lower()
            path = os.path.join(current_working_directory, 'experiments',
                                 self.experiment_name)
            os.makedirs(path, exist_ok=True)
            self._update_run_data(path)
            statistics = self.stats()
            self._write_to_disk(statistics, path)
        # TODO do a better job at catching precise exceptions
        except Exception as ex:
            logger.error(ex)
            logger.error('Could Not write experiment to to file')

    def info(self):
        data = self.stats()
        return self._write_to_screen(data)

    def stats(self, statistics=None):
        """ Stores to current working directory under filename"""
        # TODO for now silently creates a file a directory if None
        stats = {
            'experiment_start' : self.experiment_start,
            'experiment_id' : self.experiment_id,
            'experiment_status' : self.experiment_status,
            'task_type' : self.task_type
        }
    
        if self.computed_scores:
            stats['scores'] = self.computed_scores
        elif self.metrics and self.runs:
            stats['scores'] = self._compare()

        if statistics:
            stats.update(statistics)
        stats['runs'] = self.runs
        return stats

    def _write_to_screen(self, stats):
        print(json.dumps(stats, 
                          default = lambda o: '<MODEL>',
                          indent=4, 
                          sort_keys=True)
             )

    def _write_to_disk(self, stats, path):
        
        for name, run in stats['runs'].items():
            model = run.get('model') 
            if model:
                joblib.dump(model, run['model_save_file'])
                # stash model so that  we can seralize the other stats
        stats_fname = os.path.join(path, 'experiment_statistics.json')
        with open(stats_fname, 'w') as _f:
            json.dump(stats, _f, default=lambda o: '<MODEL>') 
    
    def _update_run_data(self, experiment_path):
        for name, run in self.runs.items():
            save_file = os.path.join(experiment_path, name + '_run.model')
            if run.get('model'):
                # run['model_name'] = run['model'].__name__
                run['model_save_file'] =  save_file
  

class MoseyExperiment(Experiment):
   
    """ Concrete implementation for the abstract Experiment class
        Note
        ----
        The pipelines estimate step should be name `clf` as the model stats
        routine will look for that `cls` to read the data for the individual models

        The final estimator  in the pipeline are expected to named 'clf' 
    """
    pass
    
    
     
