from abc import ABC

class Executer(ABC):
    def __init__(self):
        pass

    def initialize_file(self,file_path):
        pass

    def reset_file(self,file_path):
        pass

    def execute(self, dataset, executions, file_path):
        pass