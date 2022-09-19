from abc import abstractmethod
import math

from sklearn.metrics import mean_squared_error, make_scorer

class Metrics:
    @abstractmethod
    def rmse_error():
        def rmse(y_true, y_pred):
            return math.sqrt(mean_squared_error(y_true, y_pred))
        return make_scorer(rmse, greater_is_better=False)

    @abstractmethod
    def custom_accuracy_metric():
        def accuracy(y_true, y_pred):
            accuracy_list = []
            for true_, pred in zip(y_true, y_pred):
                accuracy_list.append(0 if abs(true_-pred) > 3.0 else 1)

            return sum(accuracy_list) / len(y_true)

        return make_scorer(accuracy)
