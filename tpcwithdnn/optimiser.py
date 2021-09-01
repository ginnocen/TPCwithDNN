"""
Common optimizer interface, as used in steer_analysis.py
"""

from tpcwithdnn import plot_utils

class Optimiser:
    """
    The abstract top class for distortion calibration.
    """
    def __init__(self, config):
        """
        Initialize the optimizer

        :param CommonSettings config: a singleton settings object
        """
        self.config = config

    def train(self):
        """
        Train the optimizer.

        :raises NotImplementedError: it is a base class that should not be directly used
        """
        raise NotImplementedError("Calling empty train method in abstract base optimiser class")

    def apply(self):
        """
        Apply the optimizer.

        :raises NotImplementedError: it is a base class that should not be directly used
        """
        raise NotImplementedError("Calling empty apply method in abstract base optimiser class")

    def plot(self):
        """
        Plot apply results.
        """
        plot_utils.plot(self.config)

    def draw_profile(self, events_counts):
        """
        Draw the profiles of mean error, std dev and both vs the expected distortion.

        :param list events_counts: list of tuples (train_events, val_events, apply_events,
                                   total_events) with the numbers of events used
        """
        plot_utils.draw_mean(self.config, events_counts)
        plot_utils.draw_std_dev(self.config, events_counts)
        plot_utils.draw_mean_std_dev(self.config, events_counts)

    def search_grid(self):
        """
        Perform grid search to find the best model configuration.

        :raises NotImplementedError: it is a base class that should not be directly used
        """
        raise NotImplementedError("Calling empty search grid method in base optimiser class")

    def bayes_optimise(self):
        """
        Perform Bayesian optimization to find the best model configuration.

        :raises NotImplementedError: it is a base class that should not be directly used
        """
        raise NotImplementedError("Calling empty Bayes optimise method in base optimiser class")

    def save_model(self, model):
        """
        Save the model to a file.

        :param obj model: the model instance to be saved
        :raises NotImplementedError: it is a base class that should not be directly used
        """
        raise NotImplementedError("Calling empty save model method in abstract optimiser class")

    def load_model(self):
        """
        Load the model from a file.

        :return: the loaded model
        :rtype: obj
        :raises NotImplementedError: it is a base class that should not be directly used
        """
        raise NotImplementedError("Calling empty load model method in abstract optimiser class")
