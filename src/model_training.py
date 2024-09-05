import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model



def save_model(model, filepath):
    import joblib
    joblib.dump(model, filepath)

if __name__ == "__main__":
    featured_data_path = '../data/processed/featured_data.csv'
    
    data = pd.read_csv(featured_data_path)
    X_train, X_test, y_train, y_test = train_test_split(data.drop('Class', axis=1), data['Class'], test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    save_model(model, '../models/random_forest_model.pkl')
    print("Model trained and saved.")
