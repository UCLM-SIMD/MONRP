import unittest

import numpy as np
from algorithms.GRASP.GRASP import GRASP as tested_algorithm_class
from algorithms.GRASP.GraspSolution import GraspSolution


class GRASPTestCase(unittest.TestCase):

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
        self.algorithm.solutions_per_iteration = 5
        self.algorithm.iterations = 5

        result = self.algorithm.run()

        expected_num_generations = 5
        expected_num_evaluations = 125
        expected_pop_size = 1

        actual_population = result["population"]
        actual_num_generations = result["numGenerations"]
        actual_num_evaluations = result["numEvaluations"]
        actual_pop_size = len(result["population"])

        self.assertEqual(actual_num_generations, expected_num_generations)
        self.assertEqual(actual_num_evaluations, expected_num_evaluations)
        self.assertEqual(actual_pop_size, expected_pop_size)

        expected_genes = [[1, 1, 1, 1, 1]]
        for i in range(expected_pop_size):
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_genes[i], actual_population[i].selected))
