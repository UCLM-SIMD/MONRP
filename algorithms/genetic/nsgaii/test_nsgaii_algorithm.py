import unittest

import numpy as np
from algorithms.genetic.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm as tested_algorithm_class


class NSGAIITestCase(unittest.TestCase):

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
        expected_num_evaluations = 55  # 5+ 5*(5+5) =55;
        expected_pop_size = 5

        actual_population = result["population"]
        actual_num_generations = result["numGenerations"]
        actual_num_evaluations = result["numEvaluations"]
        actual_pop_size = len(result["population"])

        self.assertEqual(actual_num_generations, expected_num_generations)
        self.assertEqual(actual_num_evaluations, expected_num_evaluations)
        self.assertEqual(actual_pop_size, expected_pop_size)

        expected_genes = [[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 1],
                          [1, 1, 1, 1, 1]]

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

        expected_genes = [[0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1],
                          [0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1],
                          [0, 0, 0, 0, 1]]

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

        expected_genes = [[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 1],
                          [1, 1, 1, 1, 1]]

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
                          [1, 0, 0, 0, 0]]

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

        expected_genes = [[0, 0, 0, 0, 1],
                          [0, 1, 0, 0, 0],
                          [0, 1, 0, 0, 1],
                          [1, 0, 1, 0, 1],
                          [1, 0, 1, 1, 1]]

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

        expected_genes = [[1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0]]

        for i in range(len(expected_genes)):
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_genes[i], actual_population[i].selected))

    def test_fast_nondominated_sort(self):
        """
        Test that `fast_nondominated_sort()` method works
        """
        self.algorithm.population_length = 5
        self.algorithm.max_generations = 5
        self.algorithm.selection_candidates = 2

        result = self.algorithm.run()

        actual_population, actual_fronts = self.algorithm.fast_nondominated_sort(
            result["population"])

        expected_population = [[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1],
                               [1, 0, 0, 0, 1],
                               [1, 1, 1, 1, 1]]

        for i in range(len(actual_population)):
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_population[i], actual_population[i].selected))

        expected_fronts_appended = [[0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1]]

        expected_fronts_len = 2
        actual_fronts_len = len(actual_fronts)

        self.assertEqual(expected_fronts_len, actual_fronts_len)

        actual_fronts_appended = actual_fronts[0]+actual_fronts[1]

        for i in range(len(actual_fronts_appended)):
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_fronts_appended[i], actual_fronts_appended[i].selected))

        expected_ranks = [0, 0, 0, 0, 0]
        expected_domination_count = [0, 0, 0, 0, 0]

        for i in range(len(actual_fronts_appended)):
            print(actual_fronts_appended[i].rank)
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_ranks[i], actual_fronts_appended[i].rank))
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_domination_count[i], actual_fronts_appended[i].domination_count))

    def test_calculate_crowding_distance(self):
        """
        Test that `calculate_crowding_distance()` method works
        """
        self.algorithm.population_length = 5
        self.algorithm.max_generations = 5
        self.algorithm.selection_candidates = 2

        result = self.algorithm.run()

        actual_population = self.algorithm.calculate_crowding_distance(
            result["population"])

        expected_crowding_distance = [
            float("inf"), 0.64, 1.17, 1.36, float("inf")]

        actual_crowding_distance = [
            x.crowding_distance for x in actual_population]
        actual_crowding_distance = np.around(actual_crowding_distance, 2)

        for i in range(len(actual_population)):
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_crowding_distance[i], actual_crowding_distance[i]))

    def test_crowding_operator(self):
        """
        Test that `crowding_operator()` method works
        """
        self.algorithm.population_length = 5
        self.algorithm.max_generations = 5
        self.algorithm.selection_candidates = 2

        result = self.algorithm.run()

        actual_result1 = self.algorithm.crowding_operator(
            result["population"][0], result["population"][1])

        expected_result1 = 1

        self.assertEqual(expected_result1, actual_result1)

        actual_result2 = self.algorithm.crowding_operator(
            result["population"][2], result["population"][3])

        expected_result2 = -1

        self.assertEqual(expected_result2, actual_result2)
