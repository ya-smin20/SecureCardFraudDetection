import pandas as pd

def feature_engineering(data):
   
    data['TransactionAmt_to_mean_card1'] = data['TransactionAmt'] / data['card1'].mean()
    data['TransactionAmt_to_std_card1'] = data['TransactionAmt'] / data['card1'].std()
    
    return data

def save_featured_data(data, filepath):
    data.to_csv(filepath, index=False)

if __name__ == "__main__":
    processed_data_path = '../data/processed/cleaned_data.csv'
    featured_data_path = '../data/processed/featured_data.csv'
    
    data = pd.read_csv(processed_data_path)
    featured_data = feature_engineering(data)
    save_featured_data(featured_data, featured_data_path)
    print(f"Featured data saved to {featured_data_path}")
