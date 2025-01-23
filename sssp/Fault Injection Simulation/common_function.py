# -*- coding: utf-8 -*-

# This script is mainly used to:
# 1. Delete unnecessary lines from dryrun.out.
# 2. Backup the PTX file.
# 3. Generate a PTX file that can be used for error injection.

ptx_file = "SSSP.ptx"
dryrun_before = "dryrun.out"  # The file that needs to be cleaned, contains all compilation commands
dryrun_file = "dryrun1.out"  # The file after removing unnecessary content, used to execute from ptxas command
back_file = "SSSP1.ptx"  # Backup of the original PTX file
temp_file = "temp.ptx"  # Temporary PTX file for error injection


# Function to delete unnecessary lines from dryrun.out
def delete_line():
    f1 = open(dryrun_before, 'r')  # Open the original dryrun.out file
    f2 = open(dryrun_file, 'w')    # Open the new file to save cleaned content

    i = 1
    for line in f1.readlines():
        # From line 20 onwards, keep content starting from ptxas, delete everything above it
        if i < 20:
            i += 1
            continue
        else:
            line1 = line.strip('#$ ')  # Remove unwanted characters
            f2.write(line1)  # Write cleaned content to the new file
            i += 1
    f1.close()
    f2.close()


# Function to backup the PTX file
def back_ptx():
    f1 = open(ptx_file, 'r')  # Open the original PTX file
    f2 = open(back_file, 'w')  # Open the backup PTX file

    for line in f1.readlines():
        f2.write(line)  # Write all lines from original PTX to the backup file

    f1.close()
    f2.close()


# Function to read a PTX kernel and select instructions for error injection
# Outputs the number of kernels and labels
def read_ptx():
    line_num = 0  # Line number tracker
    f1 = open(back_file, 'r')  # Open the backup PTX file
    f2 = open(temp_file, 'w')  # Open the temporary PTX file to write the processed content

    for line in f1.readlines():
        line_num += 1
        # Strip whitespace from each line
        analyze_line = line.strip()
        # Check if the line contains a kernel name, and write it to temp.ptx with line number
        if analyze_line.startswith('// .globl'):
            f2.write(str(line_num) + " ")  # Write the line number
            kernel_name = analyze_line.split()[-1]  # Extract the kernel name
            f2.write(kernel_name + "\n")  # Write the kernel name to the temporary file
        if analyze_line == "":
            continue  # Skip empty lines
        if analyze_line.startswith('//'):
            continue  # Skip comment lines
        if analyze_line.startswith('.'):
            continue  # Skip section directives
        if analyze_line.startswith('(') or analyze_line.startswith(')'):
            continue  # Skip parentheses
        if analyze_line.startswith('}') or analyze_line.startswith('{'):
            continue  # Skip curly braces
        if analyze_line.startswith('@'):
            continue  # Skip special instructions
        if analyze_line.startswith('ret'):
            continue  # Skip return instructions
        # Skip branch instructions that have no destination registers
        if analyze_line.startswith('bra'):
            continue
        else:
            f2.write(str(line_num))  # Write line number
            f2.write(" " + analyze_line + "\n")  # Write the instruction along with its line number
    f1.close()
    f2.close()


# Main function to orchestrate the steps
def main():
    # 1. Make keep and make dry after
    # 2. Delete unnecessary lines in dryrun.out once
    delete_line()
    print("Create dryrun1.out")
    # 3. Backup PTX file once
    back_ptx()
    print("Backup PTX file")
    # 4. Generate temp.ptx (only once for error injection)
    read_ptx()
    print("Create temp.txt")


if __name__ == "__main__":
    main()
