import pandas as pd

def feature_engineering(data):

    if 'V1' in data.columns and 'V2' in data.columns:
        data['V1_to_mean_V2'] = data['V1'] / data['V2'].mean()
        data['V1_to_std_V2'] = data['V1'] / data['V2'].std()
    else:
        raise KeyError("Columns 'V1' and/or 'V2' do not exist in the dataset")
    
    return data

def save_featured_data(data, filepath):
    data.to_csv(filepath, index=False)

if __name__ == "__main__":
    processed_data_path = 'C:/my files/SecureCardFraudDetection/data/processed/cleaned_data.csv'
    featured_data_path = 'C:/my files/SecureCardFraudDetection/data/processed/featured_data.csv'
    
    data = pd.read_csv(processed_data_path)
    featured_data = feature_engineering(data)
    save_featured_data(featured_data, featured_data_path)
    print(f"Featured data saved to {featured_data_path}")
