# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import csv

# Define the input and output file names
input_file = 'nba.txt'
output_file = 'nba_roster.csv'

# Initialize lists to store data
roster_data = []
current_team = None

# Open the input file
with open(input_file, 'r') as file:
    for line in file:
        # Remove leading and trailing whitespaces from the line
        line = line.strip()
        
        # Check if the line starts with 'https', skip it
        if line.startswith('https'):
            continue
        
        # Check if the line contains the team name
        if 'Roster' in line:
            current_team = line.split(' Roster ')[0]
        elif line and "NAME" not in line:  # Skip empty lines and header lines
            # Split the line using tabs as the delimiter
            player_info = line.split('\t')
            
            # Remove any numbers from the player's name and set Jersey accordingly
            name = ''.join([c for c in player_info[0] if not c.isdigit()])
            jersey = ''.join([c for c in player_info[0] if c.isdigit()])
            
            # If no number found, set Jersey to "NA"
            if not jersey:
                jersey = "NA"
            
            # Append the team name, name, and jersey to the player's data
            player_info = [current_team, name, jersey] + player_info[1:]
            
            # Append the player's data to the roster_data list
            roster_data.append(player_info)

# Write the data to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the header row
    writer.writerow(['Team', 'NAME', 'Jersey', 'POS', 'AGE', 'HT', 'WT', 'COLLEGE', 'SALARY'])
    
    # Write the player data
    writer.writerows(roster_data)

print(f'Conversion completed. Data saved to {output_file}')

