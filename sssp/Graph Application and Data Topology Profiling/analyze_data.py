import sys

OUTCOME_STR = ["Masked", "SDC", "DUE", "SUM"]
results_app_table = {} 
results_kernel_table = {} 
results_iter_table = {} 
results_kernel_iter_table = {} 
results_vertex_table = {}
results_iter_vertex_table = {} 
results_kernel_vertex_table = {}  

# Function to check and create nested dictionaries for storing results
def check_and_create_nested_dict(dict_name, k1, k2, k3=""):
    if k1 not in dict_name:
        dict_name[k1] = {}
    if k2 not in dict_name[k1]:
        dict_name[k1][k2] = 0 if k3 == "" else {} 
    if k3 == "":
        return
    if k3 not in dict_name[k1][k2]:
        dict_name[k1][k2][k3] = 0 

# Function to add outcomes and metrics to the result tables
def add(app, kname, iter, vertex, outcome, metric1, metric2):
    # Adding data to app level
    check_and_create_nested_dict(results_app_table, app, outcome)  # Application level
    results_app_table[app][outcome] += 1
    
    # Adding data to kernel level
    check_and_create_nested_dict(results_kernel_table, kname, outcome)  # Kernel level
    results_kernel_table[kname][outcome] += 1 
    
    # Adding data to iteration level
    check_and_create_nested_dict(results_iter_table, iter, outcome)  # Iteration level
    results_iter_table[iter][outcome] += 1 

    # Adding data to kernel-iteration level
    check_and_create_nested_dict(results_kernel_iter_table, kname, iter, outcome)
    results_kernel_iter_table[kname][iter][outcome] += 1 

    # Adding data to vertex level and specific metrics
    check_and_create_nested_dict(results_vertex_table, vertex, outcome)  # Vertex level
    check_and_create_nested_dict(results_vertex_table, vertex, "metric1")  # SDC metric1
    results_vertex_table[vertex][outcome] += 1
    if outcome == 'SDC':
        results_vertex_table[vertex]["metric1"] += metric1 

    # Adding data to kernel-vertex level
    check_and_create_nested_dict(results_kernel_vertex_table, kname, vertex, outcome) 
    results_kernel_vertex_table[kname][vertex][outcome] += 1 

    # Adding data to iteration-vertex level
    check_and_create_nested_dict(results_iter_vertex_table, iter, vertex, outcome)
    results_iter_vertex_table[iter][vertex][outcome] += 1 

# Function to parse the new outcome file
def parse_outcome(app, new_file):
    try:
        rf = open(new_file, "r")
    except IOError: 
        print("Error opening new outcome file ")
        return 

    num_lines = 0
    for line in rf: 
        words1 = line.split(",")
        # Processing the lines for different applications (SSSP, SCC, Pagerank, etc.)
        if len(words1) == 19 and ('sssp' in app.lower() or 'scc' in app.lower()):
            [kname, iter, thread, vertex, fi, outcome, metric1, metric2] = [words1[2], int(words1[3]), int(words1[4]), int(words1[5]), int(words1[15]), words1[16], int(words1[17]), float(words1[18])]
        elif len(words1) == 19 and ('hits' in app.lower() or 'pagerank' in app.lower()):
            [kname, iter, thread, vertex, fi, outcome, metric1, metric2] = [words1[2], int(words1[3]), int(words1[4]), int(words1[5]), int(words1[15]), words1[16], int(words1[17]), int(words1[18])]
        elif len(words1) == 17 and words1[-3] == 'DUE':
            [kname, iter, thread, vertex, fi, outcome, metric1, metric2] = [words1[2], int(words1[3]), int(words1[4]), int(words1[5]), 1, words1[14], int(words1[15]), int(words1[16])]
        else:
            print("Error line: " + line + " len:" + str(len(words1)))
            continue
        if thread != vertex - 1:
            print("Error line: " + line + " len:" + str(len(words1)))
            continue
        
        if fi == 1:  # Fault injection success
            if ('hits' in app.lower() or 'pagerank' in app.lower()) and outcome == 'SDC' and metric2 > 60:  # Pagerank and hits, only consider top 20 results
                outcome = 'Masked'
            add(app, kname, iter, vertex, outcome, metric1, metric2)
            num_lines += 1
                
        elif fi == 2147483647:  # Fault injection success and DUE
            outcome = "DUE"
            add(app, kname, iter, vertex, outcome, metric1, metric2)    
            num_lines += 1
            
    print(app + " data size:" + str(num_lines))
    rf.close()

# Function to process the application-level table
def process_app_table(app):
    for outcome in OUTCOME_STR:
        if outcome not in results_app_table[app]:
            results_app_table[app][outcome] = 0 
    results_app_table[app]["SUM"] = results_app_table[app]["Masked"] + results_app_table[app]["SDC"] + results_app_table[app]["DUE"]
    results_app_table[app]["Masked"] = results_app_table[app]["Masked"] / results_app_table[app]["SUM"]
    results_app_table[app]["SDC"] = results_app_table[app]["SDC"] / results_app_table[app]["SUM"]
    results_app_table[app]["DUE"] = results_app_table[app]["DUE"] / results_app_table[app]["SUM"]

# Function to process the kernel-level table
def process_kernel_table():
    for kernel in results_kernel_table: 
        for outcome in OUTCOME_STR:
            if outcome not in results_kernel_table[kernel]:
                results_kernel_table[kernel][outcome] = 0                         
        results_kernel_table[kernel]["SUM"] = results_kernel_table[kernel]["Masked"] + results_kernel_table[kernel]["SDC"] + results_kernel_table[kernel]["DUE"]
        results_kernel_table[kernel]["Masked"] = results_kernel_table[kernel]["Masked"] / results_kernel_table[kernel]["SUM"]
        results_kernel_table[kernel]["SDC"] = results_kernel_table[kernel]["SDC"] / results_kernel_table[kernel]["SUM"]
        results_kernel_table[kernel]["DUE"] = results_kernel_table[kernel]["DUE"] / results_kernel_table[kernel]["SUM"]

# Other similar functions for iter_table, kernel_iter_table, vertex_table, etc., are defined to process the corresponding tables

# Main function to parse outcome file and generate the summary tables
def main():
    new_file = "SSSP_GP_fi_outcome.txt"
    app = new_file.split('/')[-1].split('.')[0]

    parse_outcome(app, new_file)
    # Generating summary tables
    print_app_tsv(app)
    print_kernel_tsv(app)
    print_iter_tsv(app)
    print_kernel_iter_tsv(app)
    print_vertex_tsv(app)
    print_iter_vertex_tsv(app)
    print_kernel_vertex_tsv(app)

    print(app + " analyze_data finish")

if __name__ == "__main__":
    main()
