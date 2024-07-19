import pandas as pd
import difflib

def preprocess_data(df):
    """Function to preprocess district names"""
    df['District'] = df['District'].str.replace(' ', '').str.lower()
    return df

def find_best_match(name, options):
    """Function to find the best match using difflib"""
    matches = difflib.get_close_matches(name, options, n=1, cutoff=0.6)
    return matches[0] if matches else name

def merge_datasets(kpi1, kpi2):
    """Function to merge two DataFrames using string similarity"""
    kpi1 = preprocess_data(kpi1)
    kpi2 = preprocess_data(kpi2)

    kpi2_districts = kpi2['District'].unique()
    kpi1['District'] = kpi1['District'].apply(lambda x: find_best_match(x, kpi2_districts))

    combined_df = pd.merge(kpi1, kpi2, on='District', how='outer')
    return combined_df

def main():
    # Example data
    data_kpi1 = {
        'District': ['Kathmandu', 'Kavre palanchowk', 'Dhanusa'],
        'KPI_1': [0.8, 0.75, 0.85]
    }

    data_kpi2 = {
        'District': ['Kathmandu', 'Kavrepalanchowk', 'Dhanusha'],
        'KPI_2': [0.35, 0.65, 0.6]
    }

    # Create DataFrames
    kpi1 = pd.DataFrame(data_kpi1)
    kpi2 = pd.DataFrame(data_kpi2)

    # Merge datasets
    merged_df = merge_datasets(kpi1, kpi2)

    # Save to CSV
    merged_df.to_csv('effective_merge.csv', index=False)
    print(merged_df)

if __name__ == "__main__":
    main()
