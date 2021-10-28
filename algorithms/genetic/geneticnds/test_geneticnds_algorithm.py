import unittest

import numpy as np
from algorithms.genetic.geneticnds.geneticnds_algorithm import GeneticNDSAlgorithm as tested_algorithm_class
from algorithms.GRASP.GraspSolution import GraspSolution
import collections


class GeneticNDSTestCase(unittest.TestCase):

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
        expected_num_evaluations = 30  # 5+ 5*5 =30;      600(?)
        expected_pop_size = 4  # 7

        actual_population = result["population"]
        actual_num_generations = result["numGenerations"]
        actual_num_evaluations = result["numEvaluations"]
        actual_pop_size = len(result["population"])

        self.assertEqual(actual_num_generations, expected_num_generations)
        self.assertEqual(actual_num_evaluations, expected_num_evaluations)
        self.assertEqual(actual_pop_size, expected_pop_size)

        expected_genes = [[0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1],
                          [0, 1, 0, 0, 1],
                          [1, 1, 0, 0, 1]]

        for i in range(expected_pop_size):
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_genes[i], actual_population[i].selected))
