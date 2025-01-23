# -*- coding: utf-8 -*-

# Used to check the result of error injection

import os
import numpy as np
import sys

# File paths for different outputs
stderr_file = "stderr.txt"
stdout_file = "stdout.txt"
output_file = "PageRank.out"

diff_file = "diff.log"
golden_output = "golden/PageRank.out"
outcome_file = "outcome.txt"
iter_file = 'iter_info.txt'
outcome = 0

# Function to get the execution time from the golden GPU result file
def get_golden_time(file_path):    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if lines:
                line = lines[-1].strip()  # Get the last line
                return float(line)
            else:
                print("File is empty")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except ValueError:
        print("Last line is not a floating point number")

# Read parameters from the shell script
metric1 = 0
metric2 = 0
para = int(sys.argv[1])  # Parameter 1 from command-line argument
itera_time = int(sys.argv[2])  # Iteration time from command-line argument
DUE_flag = int(sys.argv[3])  # DUE flag from command-line argument

# Check if the stderr file exists and read its content
if os.path.exists(stderr_file):
    line_num = 0
    stdout = open(stdout_file, 'r')
    size = os.path.getsize(stderr_file)
    
    # If the stderr file is not empty, set the outcome to 1 (error)
    if size != 0:
        outcome = 1
        metric = sys.maxsize
        print("======================Error file is not empty===========================")
    
    # If the parameter is not 0, set the outcome to 1 (incorrect program execution)
    elif para != 0:
        outcome = 1
        metric = sys.maxsize
        print("=========================Program did not return 0 correctly==============================")
    
    # If the DUE flag is 1, set the outcome to 1 (timeout error)
    elif DUE_flag == 1:
        outcome = 1
        metric = sys.maxsize
        print('======================DUE: Program timeout==============================')
    
    # If the diff file is empty, it means the result is masked
    else:
        if os.path.getsize(diff_file) == 0:
            print('======================Masked==============================')
            outcome = 2
        
        # If there are differences, it is classified as an SDC (Silent Data Corruption) error
        else:
            print('~~~~~~~~~~~~~~~~~~~~~~~~SDC~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            outcome = 3
            # SDC difference: Check and count differences between the golden output and current output
            first_integer1 = 0
            first_integer2 = 0
            with open(golden_output, 'r') as file1, open(output_file, 'r') as file2:
                lines1 = file1.readlines()
                lines2 = file2.readlines()

            for i in range(0, len(lines1)):
                if i < len(lines2):
                    line1 = lines1[i].strip()
                    line2 = lines2[i].strip()
                    if line1 != line2:
                        metric1 += 1
                        if metric2 == 0:
                            metric2 = i + 1  # Store the line where the first difference occurs

# Write the outcome to the outcome file
f = open(outcome_file, 'a')
if outcome == 1:
    f.write(",{0},{1},{2}\n".format("DUE", metric1, metric2))  # DUE error outcome
elif outcome == 2:
    f.write(",{0},{1},{2}\n".format("Masked", metric1, metric2))  # Masked result outcome
else:
    f.write(",{0},{1},{2}\n".format("SDC", metric1, metric2))  # SDC error outcome
f.close()
