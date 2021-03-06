# See LICENSE file for copyright and license details.
import time
from datetime import timedelta

class TrainingHook:
    """ Base class for all trainer hooks """
    def __init__(self, trainer):
        self._trainer = trainer
        trainer.attach(self)

    def pre_training(self):
        """ Hook invoked before start training """
        pass

    def pre_epoch(self):
        """ Hook invoked before start a training epoch """
        pass

    def post_epoch(self):
        """ Hook invoked after start a training epoch """
        pass

    def post_training(self):
        """ Hook invoked after training finished """
        pass

    def notify(self, msg):
        return getattr(self, msg)()

    @property
    def trainer(self):
        return self._trainer

class StatusHook(TrainingHook):
    """ Training hook for log the start and end of training
    """

    def __init__(self, trainer):
        super(StatusHook, self).__init__(trainer)

    def pre_training(self):
        print('Training started...')

    def pre_epoch(self):
        print('\tEpoch {} started...'.format(self._trainer.epoch+1))

    def post_epoch(self):
        print('\tEpoch {} finished.'.format(self._trainer.epoch))

    def post_training(self):
        print('Training ended.')

class ErrorHook(TrainingHook):
    """ Training Hook for log error at each epoch
    """

    def __init__(self, trainer,
                 error_calculator,
                 measure_before_training):
        """ Constructor

          Args:
            trainer (Trainer):              Trainer instance
            error_calculator (callable):    Error calculating procedure
            measure_before_training (bool): True for calculate error before start training
        """
        super(ErrorHook, self).__init__(trainer)
        self.measure_before_training = measure_before_training
        self.error_calculator = error_calculator
        self._errors = []

    @property
    def errors(self):
        return self._errors

    def pre_training(self):
        if self.measure_before_training:
            print("Error before training: {}.".format(self.error_calculator()))

    def post_epoch(self):
        error = self.error_calculator()
        self._errors.append(error)
        print("Epoch {} error: {}.".format(self._trainer.epoch, error))

class TimingHook(TrainingHook):
    """ Training Hook for log the training time between epochs
    """

    def __init__(self, trainer):
        super(TimingHook, self).__init__(trainer)
        self._epoch_times = []

    @property
    def epoch_times(self):
        return self._epoch_times

    @staticmethod
    def time_format(elapsed_seconds):
        return str(timedelta(seconds=elapsed_seconds))

    def mean_error(self):
        return sum(self._epoch_times)/len(self._epoch_times)

    def pre_epoch(self):
        self.start_time = time.time()

    def post_epoch(self):
        self.end_time = time.time()
        elapsed_seconds = self.end_time - self.start_time
        self._epoch_times.append(elapsed_seconds)

        eta = self.mean_error() * self._trainer._remaining_epochs

        print("Epoch {} training time: {}".format(self._trainer.epoch, self.time_format(elapsed_seconds)))
        print("ETA: {}".format(self.time_format(eta)))
