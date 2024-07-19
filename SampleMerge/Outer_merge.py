import pandas as pd

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

# Normalized district names
kpi1['District'] = kpi1['District'].str.replace(' ', '').str.lower()
kpi2['District'] = kpi2['District'].str.replace(' ', '').str.lower()

# Replacing 'Dhanusa' with 'Dhanusha' in kpi1
kpi1['District'] = kpi1['District'].replace('dhanusa', 'dhanusha')

# Merge DataFrames on 'District'
merged_df = pd.merge(kpi1, kpi2, on='District', how='outer')

# Dropped duplicate rows based on 'District'
merged_df = merged_df.drop_duplicates(subset=['District'], keep='first')

#saved to CSV
merged_df.to_csv('merged_file.csv', index=False)
print(merged_df)
