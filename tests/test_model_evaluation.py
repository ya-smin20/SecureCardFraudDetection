import unittest
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from src.model_evaluation import evaluate_model

class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        # Load data and split into train and test sets
        featured_data_path = 'C:/my files/SecureCardFraudDetection/data/processed/featured_data.csv'
        model_path = 'C:/my files/SecureCardFraudDetection/models/random_forest_model.pkl'
        
        data = pd.read_csv(featured_data_path)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.drop('Class', axis=1), data['Class'], test_size=0.2, random_state=42)
        
        self.model = joblib.load(model_path)
    
    def test_evaluate_model(self):
        # Test the evaluate_model function
        evaluate_model(self.model, self.X_test, self.y_test)

if __name__ == '__main__':
    unittest.main()
