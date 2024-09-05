import unittest
import pandas as pd
from src.data_preprocessing import split_data, clean_data

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'Feature1': [1, 2, 3, 4],
            'Feature2': [5, 6, 7, 8],
            'Class': [0, 1, 0, 1]
        })
        self.cleaned_data = clean_data(self.sample_data)
    
    def test_split_data(self):
        X_train, X_test, y_train, y_test = split_data(self.cleaned_data)
        # Ensure that the sum of the train and test sets equals the original data size
        self.assertEqual(len(X_train) + len(X_test), len(self.cleaned_data), "Total rows after splitting should match the original data size.")
        # Ensure that the number of labels matches the number of samples
        self.assertEqual(len(X_train), len(y_train), "Number of training samples should match the number of training labels.")
        self.assertEqual(len(X_test), len(y_test), "Number of testing samples should match the number of testing labels.")

if __name__ == '__main__':
    unittest.main()
