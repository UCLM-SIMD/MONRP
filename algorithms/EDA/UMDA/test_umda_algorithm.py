import unittest

import numpy as np
from algorithms.EDA.UMDA.umda_algorithm import UMDAAlgorithm as tested_algorithm_class
from models.Solution import Solution


class UMDATestCase(unittest.TestCase):

    def setUp(self):
        """  
        Set up algorithm and random seed
        """
        seed = 0
        self.algorithm = tested_algorithm_class()
        self.algorithm.set_seed(seed)

    def test_run(self):
        """  
        Test that `run()` method works
        """
        self.algorithm.population_length = 5
        self.algorithm.max_generations = 5

        result = self.algorithm.run()

        expected_num_generations = 5
        expected_num_evaluations = 30
        expected_pop_size = 5

        actual_population = result["population"]
        actual_num_generations = result["numGenerations"]
        actual_num_evaluations = result["numEvaluations"]
        actual_pop_size = len(result["population"])

        self.assertEqual(actual_num_generations, expected_num_generations)
        self.assertEqual(actual_num_evaluations, expected_num_evaluations)
        self.assertEqual(actual_pop_size, expected_pop_size)

        expected_genes = [[1, 1, 1, 0, 1], [0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 1, 0, 0, 1]]
        for i in range(expected_pop_size):
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_genes[i], actual_population[i].selected))

    def test_learn_probability_model(self):
        """
        Test that `learn_probability_model()` method works
        """
        self.algorithm.population_length = 5

        sols = [Solution(self.algorithm.dataset, None,
                         selected=[0, 1, 0, 0, 0]),
                Solution(self.algorithm.dataset, None,
                         selected=[0, 0, 0, 0, 0]),
                Solution(self.algorithm.dataset, None,
                         selected=[0, 0, 0, 0, 1]),
                Solution(self.algorithm.dataset, None,
                         selected=[0, 0, 0, 0, 1])]

        actual_probability_model = self.algorithm.learn_probability_model(sols)

        expected_probability_model = [0, 0.25, 0, 0, 0.5]

        self.assertIsNone(np.testing.assert_array_equal(
            actual_probability_model, expected_probability_model))
