import unittest
from datasets.utils.requirements_combination.combine_dataset_requirements import combine_dataset_requirements


class CombineDatasetRequirementsTestCase(unittest.TestCase):

    def _check_datasets_equal(self, expected_data, actual_data):
        self.assertEqual(expected_data["pbis_cost"], actual_data["pbis_cost"])
        self.assertEqual(
            expected_data["stakeholders_importances"], actual_data["stakeholders_importances"])
        self.assertEqual(expected_data["stakeholders_pbis_priorities"],
                         actual_data["stakeholders_pbis_priorities"])
        self.assertEqual(
            expected_data["dependencies"], actual_data["dependencies"])

    def test_run_three_combined(self):
        """ r0<->r1; r0<->r3; r1<->r2; --> r0+r1+r2+r3 """
        expected_data = {
            "pbis_cost": [3, 12],
            "stakeholders_importances": [1, 3],
            "stakeholders_pbis_priorities": [
                [5, 4],
                [5, 4]
            ],
            "dependencies": [None, None]
        }

        json_data = {
            "pbis_cost": [3, 3, 3, 3, 3],
            "stakeholders_importances": [1, 3],
            "stakeholders_pbis_priorities": [
                [4, 2, 1, 2, 5],
                [4, 2, 1, 2, 5]
            ],
            "dependencies": [[1, 3], [0, 2], [1], [0], None]
        }

        actual_data = combine_dataset_requirements(json_data)

        self._check_datasets_equal(expected_data, actual_data)

    def test_run_three_and_two_combined(self):
        """ r0<->r1; r0<->r3; r1<->r2; r4<->r5 --> r0+r1+r2+r3; r4+r5 """
        expected_data = {
            "pbis_cost": [12, 7],
            "stakeholders_importances": [1, 3],
            "stakeholders_pbis_priorities": [
                [4, 5],
                [4, 5]
            ],
            "dependencies": [None, None]
        }

        json_data = {
            "pbis_cost": [3, 3, 3, 3, 3, 4],
            "stakeholders_importances": [1, 3],
            "stakeholders_pbis_priorities": [
                [4, 2, 1, 2, 5, 4],
                [4, 2, 1, 2, 5, 3]
            ],
            "dependencies": [[1, 3], [0, 2], [1], [0], [5], [4]]
        }

        actual_data = combine_dataset_requirements(json_data)

        self._check_datasets_equal(expected_data, actual_data)

    def test_run_three_and_two_combined_unordered_dependencies(self):
        """ r0<->r3; r1<->r2; r2<->r3; r4<->r5 --> r0+r3+r1+r2; r4+r5 """
        expected_data = {
            "pbis_cost": [12, 7],
            "stakeholders_importances": [1, 3],
            "stakeholders_pbis_priorities": [
                [4, 5],
                [4, 5]
            ],
            "dependencies": [None, None]
        }

        json_data = {
            "pbis_cost": [3, 3, 3, 3, 3, 4],
            "stakeholders_importances": [1, 3],
            "stakeholders_pbis_priorities": [
                [4, 2, 1, 2, 5, 4],
                [4, 2, 1, 2, 5, 3]
            ],
            "dependencies": [[3], [2], [1, 3], [0, 2], [5], [4]]
        }

        actual_data = combine_dataset_requirements(json_data)

        self._check_datasets_equal(expected_data, actual_data)

    def test_two_combined_with_one_dependent(self):
        """ r0<->r1; r1->r2 --> r0+r1->r2 """
        expected_data = {
            "pbis_cost": [3, 3, 3, 4, 6],
            "stakeholders_importances": [1, 3],
            "stakeholders_pbis_priorities": [
                [1, 2, 5, 4, 4],
                [1, 2, 5, 3, 4]
            ],
            "dependencies": [None, None, None, None, [0]]
        }

        json_data = {
            "pbis_cost": [3, 3, 3, 3, 3, 4],
            "stakeholders_importances": [1, 3],
            "stakeholders_pbis_priorities": [
                [4, 2, 1, 2, 5, 4],
                [4, 2, 1, 2, 5, 3]
            ],
            "dependencies": [[1], [0, 2], None, None, None, None]
        }

        actual_data = combine_dataset_requirements(json_data)

        self._check_datasets_equal(expected_data, actual_data)

    def test_no_combination(self):
        """ r0->r1; r1->r2; r2->r3; r3->r4; r4->r5; r5->r0 """
        expected_data = {
            "pbis_cost": [3, 3, 3, 3, 3, 4],
            "stakeholders_importances": [1, 3],
            "stakeholders_pbis_priorities": [
                [4, 2, 1, 2, 5, 4],
                [4, 2, 1, 2, 5, 3]
            ],
            "dependencies": [[1], [2], [3], [4], [5], [0]]
        }

        json_data = {
            "pbis_cost": [3, 3, 3, 3, 3, 4],
            "stakeholders_importances": [1, 3],
            "stakeholders_pbis_priorities": [
                [4, 2, 1, 2, 5, 4],
                [4, 2, 1, 2, 5, 3]
            ],
            "dependencies": [[1], [2], [3], [4], [5], [0]]
        }

        actual_data = combine_dataset_requirements(json_data)

        self._check_datasets_equal(expected_data, actual_data)

    def test_combination_keeps_other_dependencies(self):
        """ r0->r1; r1->r2; r2->r3; r3->r4; r4->r5; r5<->r0 --> r0+r5->r1; r1->...->r4; r4->r0+r5 """
        expected_data = {
            "pbis_cost": [3, 3, 3, 3, 7],
            "stakeholders_importances": [1, 3],
            "stakeholders_pbis_priorities": [
                [2, 1, 2, 5, 4],
                [2, 1, 2, 5, 4]
            ],
            "dependencies": [[1], [2], [3], [4], [0]]
        }

        json_data = {
            "pbis_cost": [3, 3, 3, 3, 3, 4],
            "stakeholders_importances": [1, 3],
            "stakeholders_pbis_priorities": [
                [4, 2, 1, 2, 5, 4],
                [4, 2, 1, 2, 5, 3]
            ],
            "dependencies": [[1, 5], [2], [3], [4], [5], [0]]
        }

        actual_data = combine_dataset_requirements(json_data)

        self._check_datasets_equal(expected_data, actual_data)
