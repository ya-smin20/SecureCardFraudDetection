import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def clean_data(data):
    # Handling missing values
    data = data.dropna()
    
    # Removing duplicates
    data = data.drop_duplicates()
    
    return data

def split_data(data, test_size=0.2, random_state=42):
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def save_cleaned_data(data, filepath):
    data.to_csv(filepath, index=False)

if __name__ == "__main__":
    raw_data_path = 'C:/my files/SecureCardFraudDetection/data/raw/creditcard.csv'
    processed_data_path = 'C:/my files/SecureCardFraudDetection/data/processed/cleaned_data.csv'
    
    data = load_data(raw_data_path)
    cleaned_data = clean_data(data)
    save_cleaned_data(cleaned_data, processed_data_path)
    print(f"Cleaned data saved to {processed_data_path}")
