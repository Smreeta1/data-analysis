import requests
import csv
import pandas as pd  
from parse.parse import parse_line

# Function to fetch data from the provided URL
def fetch_data(url):
    try:
        
        response = requests.get(url)
        response.raise_for_status()  # HTTPError for bad responses
        content = response.text.strip()
        
        # Save the fetched content to a file
        with open('fetched.csv', mode='w', newline='') as file:
            file.write(content)  # Write the content to the file
            
        print("Data saved to fetched.csv")
        return content  # Return the content for further processing
    except requests.RequestException as e:
        raise RuntimeError(f"Error fetching data: {e}")
    except IOError as e:
        raise RuntimeError(f"Error writing to file: {e}")


# Function to process the fetched content
def process_data(content):
    lines = content.split('\n')  # Split the content into lines
    return process_lines(lines)  # Process each line and collect results


# Function to process each line and call parse_line for each one
def process_lines(lines):
    results = []            # Initialize an empty list to store parsed results
    
    for line in lines:
        try:
            result = parse_line(line)  # Parse each line
            if result:
                results.extend(result)  # Collect results from each line
        except ValueError as e:
            print(e)  # Print error message if FIPS code is not found
    
    return results


# Function to save results to a CSV file
def save_to_csv(filename, data):
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['State Name', 'State Abbr', 'FIPS Code'])  # Write the header
            writer.writerows(data)  # Write the data rows
    except IOError as e:
        raise RuntimeError(f"Error writing to file: {e}")
    

# Function to display the data in a pandas DataFrame
def display_in_dataframe(data):
    df = pd.DataFrame(data, columns=['State Name', 'State Abbr', 'FIPS Code'])
    print(df.to_string(index=False))  # Print the DataFrame without row indices

