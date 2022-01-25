import unittest

import numpy as np
from algorithms.EDA.bivariate.MIMIC.mimic_algorithm import MIMICAlgorithm as tested_algorithm_class
from models.Solution import Solution


class MIMICTestCase(unittest.TestCase):

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
                          [0, 1, 1, 0, 1], [0, 1, 0, 0, 1], [0, 0, 0, 0, 1]]
        for i in range(expected_pop_size):
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_genes[i], actual_population[i].selected))

    def test_learn_marginals(self):
        """  
        Test that `learn_marginals()` method works
        """
        sample = [0, 0, 0, 0, 1]
        sol = Solution(self.algorithm.dataset, None, selected=sample)
        sample2 = [0, 0, 0, 0, 0]
        sol2 = Solution(self.algorithm.dataset, None, selected=sample2)

        expected_marginals = [0, 0, 0, 0, 0.67]
        solutions = [sol2, sol, sol]
        actual_marginals = self.algorithm.learn_marginals(
            solutions, len(solutions))
        actual_marginals = np.around(actual_marginals, 2)
        self.assertEqual(sorted(actual_marginals),
                         sorted(expected_marginals))

        expected_marginals = [0, 0, 0, 0, 0]
        solutions = [sol2, sol2, sol2]
        actual_marginals = self.algorithm.learn_marginals(
            solutions, len(solutions))
        actual_marginals = np.around(actual_marginals, 2)
        self.assertEqual(sorted(actual_marginals),
                         sorted(expected_marginals))

    def test_get_probability_distribution(self):
        """  
        Test that `get_probability_distribution()` method works
        """
        sample = [0, 0, 0, 0, 1]
        sol = Solution(self.algorithm.dataset, None, selected=sample)
        sample2 = [0, 0, 0, 0, 0]
        sol2 = Solution(self.algorithm.dataset, None, selected=sample2)

        expected_probability_distribution = [0.33, 0.67]
        solutions = [sol2, sol, sol]
        index = 4
        actual_probability_distribution = self.algorithm.get_probability_distribution(
            solutions, index, len(solutions), laplace=0)
        actual_probability_distribution = np.around(
            actual_probability_distribution, 2)
        self.assertEqual(sorted(actual_probability_distribution),
                         sorted(expected_probability_distribution))

        expected_probability_distribution = [2.67, 3]
        actual_probability_distribution = self.algorithm.get_probability_distribution(
            solutions, index, len(solutions), laplace=1)
        actual_probability_distribution = np.around(
            actual_probability_distribution, 2)
        self.assertEqual(sorted(actual_probability_distribution),
                         sorted(expected_probability_distribution))

    def test_get_entropy(self):
        """  
        Test that `get_entropy()` method works
        """
        sample = [0, 0, 0, 0, 1]
        sol = Solution(self.algorithm.dataset, None, selected=sample)
        sample2 = [0, 0, 0, 0, 0]
        sol2 = Solution(self.algorithm.dataset, None, selected=sample2)

        expected_entropy = 0.9183
        solutions = [sol2, sol, sol]
        index = 4
        actual_entropy = self.algorithm.get_entropy(
            solutions, index, len(solutions))
        actual_entropy = round(actual_entropy, 4)
        self.assertEqual(actual_entropy, expected_entropy)

        expected_entropy = 0.0
        index = 0
        actual_entropy = self.algorithm.get_entropy(
            solutions, index, len(solutions))
        actual_entropy = round(actual_entropy, 4)
        self.assertEqual(actual_entropy, expected_entropy)

    def test_get_conditional_entropy(self):
        """  
        Test that `get_conditional_entropy()` method works
        """
        sample = [0, 0, 0, 0, 1]
        sol = Solution(self.algorithm.dataset, None, selected=sample)
        sample2 = [0, 0, 0, 0, 0]
        sol2 = Solution(self.algorithm.dataset, None, selected=sample2)

        expected_entropy = 0.8541
        solutions = [sol2, sol, sol]
        index = 0
        index2 = 4
        actual_entropy = self.algorithm.get_conditional_entropy(
            solutions, index, index2, len(solutions))
        actual_entropy = round(actual_entropy, 4)
        self.assertEqual(actual_entropy, expected_entropy)

        expected_entropy = 0.7775
        index = 0
        index2 = 2
        actual_entropy = self.algorithm.get_conditional_entropy(
            solutions, index, index2, len(solutions))
        actual_entropy = round(actual_entropy, 4)
        self.assertEqual(actual_entropy, expected_entropy)

    def test_get_lower_conditional_entropy(self):
        """  
        Test that `get_lower_conditional_entropy()` method works
        """

        sample = [0, 0, 0, 0, 1]
        sol = Solution(self.algorithm.dataset, None, selected=sample)
        sample2 = [0, 0, 0, 0, 0]
        sol2 = Solution(self.algorithm.dataset, None, selected=sample2)
        solutions = [sol2, sol, sol]

        used = [True, False, False, False, False]
        parent = 0
        expected_index = 1
        actual_index = self.algorithm.get_lower_conditional_entropy(
            solutions, parent, used, len(solutions))
        self.assertEqual(actual_index, expected_index)

        used = [True, True, False, False, True]
        parent = 4
        expected_index = 2
        actual_index = self.algorithm.get_lower_conditional_entropy(
            solutions, parent, used, len(solutions))
        self.assertEqual(actual_index, expected_index)

    def test_get_distributions(self):
        """  
        Test that `get_distributions()` method works
        """

        sample = [0, 0, 0, 0, 1]
        sol = Solution(self.algorithm.dataset, None, selected=sample)
        sample2 = [0, 0, 0, 0, 0]
        sol2 = Solution(self.algorithm.dataset, None, selected=sample2)
        solutions = [sol2, sol, sol]

        x = 0
        y = 4
        expected_prob_x = [0.8, 0.2]
        expected_prob_y = [0.4, 0.6]
        expected_prob_xy = [[0.67, 0.75], [0.33, 0.25]]
        actual_prob_x, actual_prob_y, actual_prob_xy = self.algorithm.get_distributions(
            solutions, x, y, len(solutions))
        actual_prob_x = np.around(actual_prob_x, 2)
        actual_prob_y = np.around(actual_prob_y, 2)
        actual_prob_xy = actual_prob_xy.round(2)
        self.assertIsNone(np.testing.assert_array_equal(
            actual_prob_x, expected_prob_x))
        self.assertIsNone(np.testing.assert_array_equal(
            actual_prob_y, expected_prob_y))
        self.assertIsNone(np.testing.assert_array_equal(
            actual_prob_xy, expected_prob_xy))

        x = 0
        y = 2
        expected_prob_x = [0.8, 0.2]
        expected_prob_y = [0.8, 0.2]
        expected_prob_xy = [[0.8, 0.5], [0.2, 0.5]]
        actual_prob_x, actual_prob_y, actual_prob_xy = self.algorithm.get_distributions(
            solutions, x, y, len(solutions))
        actual_prob_x = np.around(actual_prob_x, 2)
        actual_prob_y = np.around(actual_prob_y, 2)
        actual_prob_xy = actual_prob_xy.round(2)
        self.assertIsNone(np.testing.assert_array_equal(
            actual_prob_x, expected_prob_x))
        self.assertIsNone(np.testing.assert_array_equal(
            actual_prob_y, expected_prob_y))
        self.assertIsNone(np.testing.assert_array_equal(
            actual_prob_xy, expected_prob_xy))

    def test_sample_new_population(self):
        """  
        Test that `sample_new_population()` method works
        """
        self.algorithm.population_length = 5
        # sample=[0,0,0,0,1]
        # sample2=[0,0,0,0,0]
        # sols=[sol2, sol2, sol]
        marginals = [0, 0, 0, 0, 0.67]
        parents = [-1, 0, 1, 2, 3]
        variables = [0, 1, 2, 3, 4]
        conditionals = [[0., 0.],
                        [0.2, 0.5],
                        [0.2, 0.5],
                        [0.2, 0.5],
                        [0.6, 0.5]]
        expected_genes = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [
            0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]
        actual_population = self.algorithm.sample_new_population(
            marginals, parents, variables, conditionals)

        for i in range(len(expected_genes)):
            with self.subTest(i=i):
                self.assertIsNone(np.testing.assert_array_equal(
                    expected_genes[i], actual_population[i].selected))

    def test_generate_sample(self):
        """  
        Test that `generate_sample()` method works
        """
        # sample=[0,0,0,0,1]
        # sample2=[0,0,0,0,0]
        # sols=[sol2, sol2, sol]
        marginals = [0, 0, 0, 0, 0.67]
        parents = [-1, 0, 1, 2, 3]
        variables = [0, 1, 2, 3, 4]
        conditionals = [[0., 0.],
                        [0.2, 0.5],
                        [0.2, 0.5],
                        [0.2, 0.5],
                        [0.6, 0.5]]
        expected_genes = [0, 0, 0, 0, 1]
        actual_individual = self.algorithm.generate_sample(
            marginals, parents, variables, conditionals)

        self.assertIsNone(np.testing.assert_array_equal(
            expected_genes, actual_individual.selected))

    def test_learn_probability_model(self):
        """  
        Test that `learn_probability_model()` method works
        """
        sample = [0, 0, 0, 0, 1]
        sample2 = [0, 0, 0, 0, 0]
        sol = Solution(self.algorithm.dataset, None, selected=sample)
        sol2 = Solution(self.algorithm.dataset, None, selected=sample2)
        sols = [sol2, sol2, sol]
        expected_marginals = [0, 0, 0, 0, 0.33]
        expected_parents = [-1, 0, 1, 2, 3]
        expected_variables = [0, 1, 2, 3, 4]
        expected_conditionals = [[0., 0.],
                                 [0.2, 0.5],
                                 [0.2, 0.5],
                                 [0.2, 0.5],
                                 [0.4, 0.5]]
        actual_marginals, actual_parents, actual_variables, actual_conditionals = self.algorithm.learn_probability_model(
            sols, len(sols))
        actual_marginals = np.around(actual_marginals, 2)
        actual_conditionals = actual_conditionals.round(2)

        self.assertIsNone(np.testing.assert_array_equal(
            expected_marginals, actual_marginals))
        self.assertIsNone(np.testing.assert_array_equal(
            expected_parents, actual_parents))
        self.assertIsNone(np.testing.assert_array_equal(
            expected_variables, actual_variables))
        self.assertIsNone(np.testing.assert_array_equal(
            expected_conditionals, actual_conditionals))
