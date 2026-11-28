import unittest

import torch
import torch.nn.functional as F

from loss import EntityRTDETRCriterion


class TextCardinalityBoundaryLossTest(unittest.TestCase):
    def setUp(self):
        self.criterion = EntityRTDETRCriterion(
            text_cardinality_boundary_coef=0.1,
            text_cardinality_boundary_margin=0.2,
            text_presence_threshold=0.5,
        )

    def calculate_loss(self, logits, text_valid, supervised=True):
        outputs = {"pred_text_presence": logits}
        targets = [
            {
                "text_supervision": torch.tensor([supervised]),
                "text_valid": torch.tensor([text_valid]),
            }
        ]
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        return self.criterion._loss_text_cardinality_boundary(
            outputs,
            targets,
            indices,
        )

    def test_two_texts_use_second_and_third_ranked_logits(self):
        logits = torch.tensor([[[2.0, 1.0, 0.0]]], requires_grad=True)
        actual = self.calculate_loss(logits, [True, True, False])
        expected = (
            F.softplus(torch.tensor(0.2 - 1.0))
            + F.softplus(torch.tensor(0.0 + 0.2))
        ) / 2 * 0.1
        self.assertTrue(torch.allclose(actual, expected))

        actual.backward()
        self.assertEqual(logits.grad[0, 0, 0].item(), 0.0)
        self.assertLess(logits.grad[0, 0, 1].item(), 0.0)
        self.assertGreater(logits.grad[0, 0, 2].item(), 0.0)

    def test_loss_is_permutation_invariant(self):
        first = self.calculate_loss(
            torch.tensor([[[1.5, -0.5, 0.3]]]),
            [True, False, False],
        )
        second = self.calculate_loss(
            torch.tensor([[[-0.5, 0.3, 1.5]]]),
            [True, False, False],
        )
        self.assertTrue(torch.allclose(first, second))

    def test_zero_and_three_text_boundaries(self):
        zero_text_loss = self.calculate_loss(
            torch.tensor([[[1.0, 0.0, -1.0]]]),
            [False, False, False],
        )
        three_text_loss = self.calculate_loss(
            torch.tensor([[[1.0, 0.0, -1.0]]]),
            [True, True, True],
        )
        expected_zero = F.softplus(torch.tensor(1.2)) * 0.1
        expected_three = F.softplus(torch.tensor(1.2)) * 0.1
        self.assertTrue(torch.allclose(zero_text_loss, expected_zero))
        self.assertTrue(torch.allclose(three_text_loss, expected_three))

    def test_unsupervised_entity_is_ignored(self):
        logits = torch.zeros((1, 1, 3), requires_grad=True)
        loss = self.calculate_loss(
            logits,
            [True, False, False],
            supervised=False,
        )
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.all(logits.grad == 0))

    def test_cardinality_groups_are_averaged_equally(self):
        logits = torch.tensor(
            [
                [
                    [2.0, -2.0, -3.0],
                    [2.0, -2.0, -3.0],
                    [2.0, 1.0, 0.0],
                ]
            ]
        )
        outputs = {"pred_text_presence": logits}
        targets = [
            {
                "text_supervision": torch.tensor([True, True, True]),
                "text_valid": torch.tensor(
                    [
                        [True, False, False],
                        [True, False, False],
                        [True, True, False],
                    ]
                ),
            }
        ]
        indices = [
            (
                torch.tensor([0, 1, 2]),
                torch.tensor([0, 1, 2]),
            )
        ]

        actual = self.criterion._loss_text_cardinality_boundary(
            outputs,
            targets,
            indices,
        )
        one_text_group = F.softplus(torch.tensor(-1.8))
        two_text_group = (
            F.softplus(torch.tensor(-0.8))
            + F.softplus(torch.tensor(0.2))
        ) / 2
        expected = (one_text_group + two_text_group) / 2 * 0.1
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
