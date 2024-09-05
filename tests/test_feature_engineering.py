import unittest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering import feature_engineering

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Example data to test with
        self.data = pd.DataFrame({
        'V1': [0.1, 0.2, 0.3],
        'V2': [1.1, 1.2, 1.3],
        'TransactionAmt': [100, 200, 300],
        'card1': [1, 2, 3]
    })
        print("Setup data:")
        print(self.data)

    def test_feature_engineering(self):
        print("\nBefore feature engineering:")
        print(self.data)

        # Call the feature engineering function
        engineered_data = feature_engineering(self.data)

        print("\nAfter feature engineering:")
        print(engineered_data)

        # Check if the new features were added
        self.assertIn('V1_to_mean_V2', engineered_data.columns)
        self.assertIn('V1_to_std_V2', engineered_data.columns)
        
        # Validate the calculation (example checks)
        self.assertAlmostEqual(engineered_data['V1_to_mean_V2'].iloc[0], 0.1 / self.data['V2'].mean())
        self.assertAlmostEqual(engineered_data['V1_to_std_V2'].iloc[0], 0.1 / self.data['V2'].std())

if __name__ == '__main__':
    unittest.main()
