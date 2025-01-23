# -*- coding: utf-8 -*-

# Used for determining the result of error injection

import os
import numpy as np
import sys

stderr_file = "stderr.txt"
stdout_file = "stdout.txt"
output_file = "HITS.out"
diff_file = "diff.log"
golden_output = "golden/HITS.out"
outcome_file = "outcome.txt"
iter_file = 'iter_info.txt'
outcome = 0

# Read program return value from shell script
metric1 = 0
metric2 = 0
para = int(sys.argv[1])  # The first argument passed to the script
itera_time = int(sys.argv[2])  # The second argument passed to the script
DUE_flag = int(sys.argv[3])  # The third argument passed to the script

# Check if the stderr file exists and is not empty
if os.path.exists(stderr_file):
    line_num = 0
    stdout = open(stdout_file, 'r')
    size = os.path.getsize(stderr_file)
    if size != 0:  # If stderr file is not empty, it indicates an error
        outcome = 1
        metric = sys.maxsize
        print("======================Error file is not empty===========================")
    elif para != 0:  # If the program didn't return the correct value (0), it indicates an error
        outcome = 1
        metric = sys.maxsize
        print("=========================Program did not return 0 correctly==============================")
    elif DUE_flag == 1:  # If the program timed out, set outcome as 1
        outcome = 1
        metric = sys.maxsize
        print('======================DUE: Program Timeout==============================')
    else:
        # If no differences are detected, it's a Masked case
        if os.path.getsize(diff_file) == 0:
            print('======================Masked==============================')
            outcome = 2
        else:
            print('~~~~~~~~~~~~~~~~~~~~~~~~SDC~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            outcome = 3  # SDC (Silent Data Corruption) detected
            # Check for differences between the golden output and the program output
            first_integer1 = 0
            first_integer2 = 0
            with open(golden_output, 'r') as file1, open(output_file, 'r') as file2:
                lines1 = file1.readlines()
                lines2 = file2.readlines()

            # Compare each line in the files
            for i in range(0, len(lines1)):
                if i < len(lines1) and i < len(lines2):
                    line1 = lines1[i].strip()
                    line2 = lines2[i].strip()
                    if line1 != line2:  # If there's a mismatch
                        metric1 += 1
                        if metric2 == 0:
                            metric2 = i + 1  # Record the first line where the difference occurs

# Log the result to the outcome file
f = open(outcome_file, 'a')
if outcome == 1:
    f.write(",{0},{1},{2}\n".format("DUE", metric1, metric2))
elif outcome == 2:
    f.write(",{0},{1},{2}\n".format("Masked", metric1, metric2))
else:
    f.write(",{0},{1},{2}\n".format("SDC", metric1, metric2))
f.close()
