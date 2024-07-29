# Function to parse individual lines to extract state names, state abbreviations, and fips codes

def parse_line(line):
    parts = line.split()  # Split line into parts based on whitespace
    
    # Find index of the first 2-digit numeric part(FIPS code) in `parts`. Defaults to None if not found.
    fips_index = next((i for i, part in enumerate(parts) if part.isdigit() and len(part) == 2), None)
    
    if fips_index is None:
        raise ValueError(f"FIPS code not found in line: {line}")

    state_name = ' '.join(parts[:fips_index - 1])
    state_abbr = parts[fips_index - 1]
    fips_code = parts[fips_index]
    
    '''
    # state_name: Joins All parts before the abbreviation. E.g., 'New York'.
    # state_abbr: Part before the FIPS code. E.g., 'NY'.
    # fips_code: The FIPS code itself. E.g., '36'.
    '''
    
    result = [[state_name, state_abbr, fips_code]]
    
    # Process the part after the FIPS code
    remaining_parts = parts[fips_index + 1:] #Parts after the FIPS code.
    if remaining_parts:
        remaining_state_name = ' '.join(remaining_parts[:-2]) #Joins all parts except the last two.
        remaining_state_abbr = remaining_parts[-2]  #Second-to-last part (abbreviation)
        remaining_fips_code = remaining_parts[-1]   #Last part, assumed FIPS code.
        result.append([remaining_state_name, remaining_state_abbr, remaining_fips_code])
    
    return result