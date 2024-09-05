import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.model_training import train_model

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        # Load your cleaned data
        cleaned_data = pd.read_csv('C:/my files/SecureCardFraudDetection/data/processed/cleaned_data.csv')
        
        # Use a small subset of data for testing
        X = cleaned_data.drop('Class', axis=1)
        y = cleaned_data['Class']
        
        # Split the data into a small train/test set for the purpose of testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Store the training data in the object for use in tests
        self.X_train = X_train
        self.y_train = y_train
        self.model = RandomForestClassifier(random_state=42)

    def test_train_model(self):
        trained_model = train_model(self.model, self.X_train, self.y_train)
        self.assertIsNotNone(trained_model, "The trained model should not be None.")
        self.assertGreater(len(trained_model.classes_), 1, "The model should be trained to classify more than one class.")

if __name__ == '__main__':
    unittest.main()
