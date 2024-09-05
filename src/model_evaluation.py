import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split

def evaluate_model(model, X_test, y_test):
    # Align X_test with the features the model was trained on
    expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X_test.columns[:model.n_features_in_]
    X_test_aligned = X_test[expected_features]
    
    print("Expected features (from model):", expected_features)
    print("Actual features in X_test:", X_test_aligned.columns)
    
    predictions = model.predict(X_test_aligned)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    featured_data_path = 'C:/my files/SecureCardFraudDetection/data/processed/featured_data.csv'
    model_path = 'C:/my files/SecureCardFraudDetection/models/random_forest_model.pkl'
    
    data = pd.read_csv(featured_data_path)
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = joblib.load(model_path)
    evaluate_model(model, X_test, y_test)
