# -*- coding: utf-8 -*-

# Used to evaluate the results of error injection

import os
import numpy as np
import sys
import math


stderr_file = "stderr.txt"  # Error log file
stdout_file = "stdout.txt"  # Standard output log file
output_file = "SSSP.out"  # Output file containing program results
diff_file = "diff.log"  # File containing differences between outputs
golden_output = "golden/SSSP.out"  # Expected "golden" output file
outcome_file = "outcome.txt"  # File to log the outcome of the test
iter_file = 'iter_info.txt'  # File for iteration information
outcome = 0  # Initial outcome set to 0

# Read program return values from shell script arguments
metric1 = 0
metric2 = 0
para = int(sys.argv[1])  # Parameter from command line
itera_time = int(sys.argv[2])  # Iteration time from command line
DUE_flag = int(sys.argv[3])  # DUE flag indicating timeout

# Check if stderr.txt file exists and has content
if os.path.exists(stderr_file):
    line_num = 0
    stdout = open(stdout_file, 'r')
    
    size = os.path.getsize(stderr_file)  # Get the size of stderr.txt
    if size != 0:
        outcome = 1  # If stderr is not empty, set outcome to 1 (indicating error)
        metric1 = sys.maxsize  # Set metric1 to max size to signal error
        metric2 = sys.maxsize  # Set metric2 to max size to signal error
        print("====================== Error file is not empty ============================")
    elif para != 0:
        outcome = 1  # If parameter is not zero, set outcome to 1 (indicating error)
        metric1 = sys.maxsize
        metric2 = sys.maxsize
        print("========================= Program did not return 0 ==============================")
    elif DUE_flag == 1:
        outcome = 1  # If DUE flag is 1, set outcome to 1 (indicating timeout)
        metric1 = sys.maxsize
        metric2 = sys.maxsize
        print('====================== DUE: Program timed out ==============================')
    else:
        # If diff_file is empty, the result is "Masked"
        if os.path.getsize(diff_file) == 0:
            print('====================== Masked ==============================')
            outcome = 2  # Set outcome to 2 for "Masked"
        else:
            # If there is a difference, it is a "SDC" (Silent Data Corruption)
            print('~~~~~~~~~~~~~~~~~~~~~~~~ SDC ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            outcome = 3  # Set outcome to 3 for "SDC"
            
            # Read the golden output file and create a vector
            with open(golden_output, 'r') as f:
                golden_vector = [int(line) for line in f]
            
            # Read the actual output file and create a vector
            with open(output_file, 'r') as f:
                output_vector = [int(line) for line in f]

            # Compare the golden and output vectors element by element
            for i in range(len(golden_vector)):
                if golden_vector[i] != output_vector[i]:
                    metric1 = metric1 + 1  # Count the number of mismatches
            
            # Calculate the relative L2 norm (used for error magnitude comparison)
            squared_diff = [(x - y) ** 2 for x, y in zip(golden_vector, output_vector)]
            golden_squared = [x ** 2 for x in golden_vector]
            metric2 = math.sqrt(sum(squared_diff) / sum(golden_squared))  # Compute the L2 norm

# Write the test outcome to the outcome file
f = open(outcome_file, 'a')
if outcome == 1:
    f.write(",{0},{1},{2}\n".format("DUE", metric1, metric2))  # If DUE, record metrics
elif outcome == 2:
    f.write(",{0},{1},{2}\n".format("Masked", metric1, metric2))  # If Masked, record metrics
else:
    f.write(",{0},{1},{2}\n".format("SDC", metric1, metric2))  # If SDC, record metrics
f.close()
