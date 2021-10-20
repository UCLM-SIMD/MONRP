from abc import ABC
import random

from algorithms.GRASP.Dataset import Dataset
import numpy as np

class Algorithm(ABC):

    def __init__(self, dataset_name:str="1", random_seed:int=None, debug_mode:bool=False, tackle_dependencies:bool=False):

        self.dataset:Dataset = Dataset(dataset_name)
        self.dataset_name:str = dataset_name

        self.debug_mode:bool = debug_mode
        self.tackle_dependencies:bool = tackle_dependencies

        self.set_seed(random_seed)
    
    def set_seed(self,seed:int):
        self.random_seed:int = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def reset(self):
        pass

    def run(self):
        pass

    def generate_chart(self, plot):
        result = self.run()
        func = [i.objectives for i in result["population"]]
        function1 = [i[0].value for i in func]
        function2 = [i[1].value for i in func]
        plot.scatter(function2, function1, label=self.get_name())
        return function1, function2

    def get_name(self):
        return "Algorithm"

    def stop_criterion(self):
        pass


