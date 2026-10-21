import unittest

from triage_rules import is_ambiguous_delivery, predict_rule_urgency


class DummyModel:
    def predict_proba(self, messages):
        # The final class is medium in DummyEncoder below.
        return [[0.1, 0.1, 0.8]]


class DummyEncoder:
    def inverse_transform(self, indices):
        labels = ["high", "low", "medium"]
        return [labels[int(index)] for index in indices]


class UrgencyRuleTests(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.encoder = DummyEncoder()

    def urgency(self, message, intent):
        return predict_rule_urgency(
            message, intent, self.model, self.encoder
        )

    def test_failed_payment_with_debit_is_high(self):
        self.assertEqual(
            self.urgency("Payment failed but amount debit ho gaya", "payment_issue"),
            "high",
        )

    def test_duplicate_payment_is_high(self):
        self.assertEqual(
            self.urgency("Payment rendu thadava debit aagiduchu", "payment_issue"),
            "high",
        )

    def test_locked_account_is_high(self):
        self.assertEqual(
            self.urgency("En account lock aagiduchu", "account_support"),
            "high",
        )

    def test_delayed_refund_is_high(self):
        self.assertEqual(
            self.urgency("Mujhe abhi tak mera refund nahi mila", "refund_request"),
            "high",
        )

    def test_password_reset_is_low(self):
        self.assertEqual(
            self.urgency("Password marandhuten reset panna help venum", "account_support"),
            "low",
        )

    def test_refund_status_is_low(self):
        self.assertEqual(
            self.urgency("En refund process status enna", "refund_request"),
            "low",
        )

    def test_regular_refund_is_medium(self):
        self.assertEqual(
            self.urgency("Enaku kaasu thirupi venum", "refund_request"),
            "medium",
        )

    def test_non_urgent_payment_problem_defaults_to_medium(self):
        self.assertEqual(
            self.urgency("En payment success aagala", "payment_issue"),
            "medium",
        )

    def test_delivery_without_timeline_needs_review(self):
        self.assertTrue(
            is_ambiguous_delivery("Bro en order innum deliver aagala", "delivery_issue")
        )
        self.assertFalse(
            is_ambiguous_delivery("En shipment romba late aagudhu", "delivery_issue")
        )

    def test_non_delivery_message_is_not_delivery_ambiguous(self):
        self.assertFalse(
            is_ambiguous_delivery("Password reset panna help venum", "account_support")
        )


if __name__ == "__main__":
    unittest.main()
