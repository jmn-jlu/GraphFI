# -*- coding: utf-8 -*-

import os
import numpy as np
import sys
import math

stderr_file = "stderr.txt"
stdout_file = "stdout.txt"
output_file = "SCC.out"
#back_output = "back_output.txt"
diff_file = "diff.log"
golden_output = "golden/SCC.out"
#back_golden = "backup_golden_output.txt"
outcome_file = "outcome.txt"
iter_file = 'iter_info.txt'
outcome = 0

metric1 = 0
metric2 = 0
para = int(sys.argv[1])
itera_time = int(sys.argv[2])
DUE_flag = int(sys.argv[3])
if os.path.exists(stderr_file):
    line_num = 0
    stdout = open(stdout_file, 'r')

    size = os.path.getsize(stderr_file)
    if size != 0:
        outcome = 1
        metric1 = sys.maxsize
        metric2 = sys.maxsize
        print("======================Error file is not empty===========================")
    elif para != 0:
        outcome = 1
        metric2 = sys.maxsize
        metric1 = sys.maxsize
        print("=========================Program did not return 0==============================")
    elif DUE_flag == 1:
        outcome = 1
        metric1 = sys.maxsize
        metric2 = sys.maxsize
        print('======================DUE==============================')
    else:
        if os.path.getsize(diff_file) == 0:
            # Masked
            print('======================Masked==============================')
            outcome = 2
        else:
            print('~~~~~~~~~~~~~~~~~~~~~~~~SDC~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            outcome = 3

            with open(golden_output, 'r') as f:
                golden_vector = [int(line) for line in f]

            with open(output_file, 'r') as f:
                output_vector = [int(line) for line in f]

            for i in range(len(golden_vector)):
                if golden_vector[i] != output_vector[i]:
                    metric1 = metric1 + 1    

            squared_diff = [(x - y) ** 2 for x, y in zip(golden_vector, output_vector)]
            golden_squared = [x ** 2 for x in golden_vector]
            metric2 = math.sqrt(sum(squared_diff) / sum(golden_squared))            

f = open(outcome_file, 'a')
if outcome == 1:
    f.write(",{0},{1},{2}\n".format("DUE", metric1, metric2))
elif outcome == 2:
    f.write(",{0},{1},{2}\n".format("Masked", metric1, metric2))
else:
    f.write(",{0},{1},{2}\n".format("SDC", metric1, metric2))
f.close()
