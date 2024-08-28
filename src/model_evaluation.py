import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    featured_data_path = '../data/processed/featured_data.csv'
    model_path = '../models/random_forest_model.pkl'
    
    data = pd.read_csv(featured_data_path)
    X_train, X_test, y_train, y_test = train_test_split(data.drop('Class', axis=1), data['Class'], test_size=0.2, random_state=42)
    
    model = joblib.load(model_path)
    evaluate_model(model, X_test, y_test)
