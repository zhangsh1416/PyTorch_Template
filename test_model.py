import unittest
import torch
from models.model import SimpleNN

class TestModel(unittest.TestCase):
    def test_model_forward(self):
        model = SimpleNN()
        test_input = torch.randn(1, 1, 28, 28)
        test_output = model(test_input)
        self.assertEqual(test_output.shape, (1, 10))

if __name__ == '__main__':
    unittest.main()
