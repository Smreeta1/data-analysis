#function to fetch the content, process it, save results, and display them
from parse.fetch_process import fetch_data,process_data,save_to_csv, display_in_dataframe
def main():
    url = 'https://pastebin.com/raw/G0VH1LpS'
    
    try:
        content = fetch_data(url)  # Fetch data from the URL
        results = process_data(content)  # Process the fetched data
        
        # Save the results to a CSV file
        save_to_csv('parsed_result.csv', results)  # Save the results to a file
        
        # Display the results in a pandas DataFrame
        display_in_dataframe(results)
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":
    main()
