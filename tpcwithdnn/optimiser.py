"""Common optimiser interface, as used in steer_analysis.py"""
# pylint: disable=missing-function-docstring, missing-class-docstring

import tpcwithdnn.plot_utils as plot_utils

class Optimiser:
    def __init__(self, config):
        self.config = config

    def train(self):
        raise NotImplementedError("Calling empty train method in abstract base optimiser class")

    def save_model_(self, model):
        raise NotImplementedError("Calling empty save model method in abstract optimiser class")

    def apply(self):
        raise NotImplementedError("Calling empty apply method in abstract base optimiser class")

    def load_model_(self):
        raise NotImplementedError("Calling empty load model method in abstract optimiser class")

    def plot(self):
        plot_utils.plot(self.config)

    def draw_profile(self, events_counts):
        plot_utils.draw_mean(self.config, events_counts)
        plot_utils.draw_std_dev(self.config, events_counts)
        plot_utils.draw_mean_std_dev(self.config, events_counts)

    def search_grid(self):
        raise NotImplementedError("Calling empty search grid method in base optimiser class")
