from abc import ABC


class Algorithm(ABC):
    def __init__(self):
        pass

    def run(self):
        pass

    def generate_chart(self,plot):
        result=self.run()
        func = [i.objectives for i in result["population"]]
        function1 = [i[0].value for i in func]
        function2 = [i[1].value for i in func]
        plot.scatter(function2, function1,label=self.get_name())

    def get_name(self):
        return "Algorithm"
