import pandas as pd
def main():

    # Read the csv files
    ds1 = pd.read_csv('file1.csv')

    ds2 = pd.read_csv('file2.csv')

    merged_ds = pd.merge(ds1, ds2, on='District', how='outer')
    print(merged_ds)

    merged_ds.to_csv('merged_file.csv', index=False)
    
if __name__ == '__main__':
    main()