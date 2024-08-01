import unittest
from data_utils import get_data_loaders


class TestDataLoader(unittest.TestCase):
    def test_data_loader(self):
        train_loader, val_loader, test_loader = get_data_loaders()
        train_data = next(iter(train_loader))
        val_data = next(iter(val_loader))
        test_data = next(iter(test_loader))

        self.assertEqual(len(train_data), 2)
        self.assertEqual(len(val_data), 2)
        self.assertEqual(len(test_data), 2)

        self.assertEqual(train_data[0].shape[1:], (1, 28, 28))
        self.assertEqual(val_data[0].shape[1:], (1, 28, 28))
        self.assertEqual(test_data[0].shape[1:], (1, 28, 28))


if __name__ == '__main__':
    unittest.main()
