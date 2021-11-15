import unittest

import numpy as np
from algorithms.genetic.geneticnds.geneticnds_algorithm import GeneticNDSAlgorithm as tested_algorithm_class


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

    def test_generate_starting_population(self):
        """
        Test that `generate_starting_population()` method works
        """
        self.algorithm.population_length = 5

        actual_population = self.algorithm.generate_starting_population()

        expected_genes = [[0, 1, 1, 0, 1],
                          [1, 1, 1, 1, 1],
                          [1, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1],
                          [0, 1, 1, 0, 0]]

        for i in range(len(expected_genes)):
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_genes[i], actual_population[i].selected))

    def test_selection_tournament(self):
        """
        Test that `selection_tournament()` method works
        """
        self.algorithm.population_length = 5
        self.algorithm.max_generations = 5
        self.algorithm.selection_candidates = 2

        result = self.algorithm.run()

        actual_population = self.algorithm.selection_tournament(
            result["population"])

        expected_genes = [[0, 1, 0, 0, 1],
                          [0, 0, 0, 0, 1],
                          [1, 1, 0, 0, 1],
                          [1, 1, 0, 0, 1]]

        for i in range(len(expected_genes)):
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_genes[i], actual_population[i].selected))

    def test_crossover_one_point(self):
        """
        Test that `crossover_one_point()` method works
        """
        self.algorithm.population_length = 5
        self.algorithm.max_generations = 5

        result = self.algorithm.run()

        actual_population = self.algorithm.crossover_one_point(
            result["population"])

        expected_genes = [[0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1],
                          [0, 1, 0, 0, 1],
                          [1, 1, 0, 0, 1]]

        for i in range(len(expected_genes)):
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_genes[i], actual_population[i].selected))

    def test_crossover_aux_one_point(self):
        """
        Test that `crossover_aux_one_point()` method works
        """
        self.algorithm.population_length = 5
        self.algorithm.max_generations = 5

        result = self.algorithm.run()

        actual_population = self.algorithm.crossover_aux_one_point(
            result["population"][0], result["population"][3])

        expected_genes = [[0, 0, 0, 0, 1],
                          [1, 1, 0, 0, 1]]

        for i in range(len(expected_genes)):
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_genes[i], actual_population[i].selected))

    def test_mutation_flip1bit(self):
        """
        Test that `mutation_flip1bit()` method works
        """
        self.algorithm.population_length = 5
        self.algorithm.max_generations = 5

        result = self.algorithm.run()

        self.algorithm.mutation_prob = 1.0
        actual_population = self.algorithm.mutation_flip1bit(
            result["population"])

        expected_genes = [[0, 1, 0, 0, 1],
                          [0, 0, 0, 1, 1],
                          [0, 1, 0, 0, 0],
                          [0, 1, 0, 0, 1]]

        for i in range(len(expected_genes)):
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_genes[i], actual_population[i].selected))

    def test_mutation_flipeachbit(self):
        """
        Test that `mutation_flipeachbit()` method works
        """
        self.algorithm.population_length = 5
        self.algorithm.max_generations = 5

        result = self.algorithm.run()

        self.algorithm.mutation_prob = 1.0
        actual_population = self.algorithm.mutation_flipeachbit(
            result["population"])

        expected_genes = [[1, 1, 1, 1, 0],
                          [1, 1, 1, 1, 0],
                          [1, 0, 1, 1, 0],
                          [0, 0, 1, 1, 0]]

        for i in range(len(expected_genes)):
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_genes[i], actual_population[i].selected))

    def test_replacement_elitism(self):
        """
        Test that `replacement_elitism()` method works
        """
        self.algorithm.population_length = 5
        self.algorithm.max_generations = 5

        result = self.algorithm.run()

        actual_population = self.algorithm.replacement_elitism(
            result["population"], result["population"])

        expected_genes = [[1, 1, 0, 0, 1],
                          [0, 0, 0, 0, 1],
                          [0, 1, 0, 0, 1],
                          [1, 1, 0, 0, 1]]

        for i in range(len(expected_genes)):
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_genes[i], actual_population[i].selected))
