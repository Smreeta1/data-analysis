import pandas as pd

# Example data
df1 = pd.DataFrame({
    'District': ['Kathmandu', 'Kavre palanchowk', 'Dhanusa'],
    'KPI_1': [0.8, 0.75, 0.85]
})

df2 = pd.DataFrame({
    'District': ['Kathmandu', 'Kavrepalanchowk', 'Dhanusha'],
    'KPI_2': [0.35, 0.65, 0.6]
})

# Outer merge
outer_merge = pd.merge(df1, df2, on='District', how='outer')
print("Outer Merge:")
print(outer_merge)
print()

# Inner merge
inner_merge = pd.merge(df1, df2, on='District', how='inner')
print("Inner Merge:")
print(inner_merge)
print()

# Left merge
left_merge = pd.merge(df1, df2, on='District', how='left')
print("Left Merge:")
print(left_merge)
print()

# Right merge
right_merge = pd.merge(df1, df2, on='District', how='right')
print("Right Merge:")
print(right_merge)

# Save each merge result to a CSV file
outer_merge.to_csv('outer_merge.csv', index=False)
inner_merge.to_csv('inner_merge.csv', index=False)
left_merge.to_csv('left_merge.csv', index=False)
right_merge.to_csv('right_merge.csv', index=False)

print("Merge results saved to CSV files.")
