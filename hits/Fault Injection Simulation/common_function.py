# -*- coding: utf-8 -*-

# This script is used to delete unnecessary lines from the dryrun.out file, backup the ptx file, and generate ptx files for fault injection.
ptx_file = "HITS.ptx"  # The original PTX file
dryrun_before = "dryrun.out"  # The file containing all compilation commands, to be cleaned up
dryrun_file = "dryrun1.out"  # The file after unnecessary content has been removed, starting from the ptxas command
back_file = "HITS1.ptx"  # The backup of the original PTX file
temp_file = "temp.ptx"  # Temporary PTX file for fault injection

# Function to delete unnecessary lines from the dryrun.out file
def delete_line():
    f1 = open(dryrun_before, 'r')
    f2 = open(dryrun_file, 'w')

    i = 1
    for line in f1.readlines():
        # Keep the content from line 20 (the ptxas command onwards)
        if i < 20:
            i += 1
            continue
        else:
            line1 = line.strip('#$ ')  # Remove unwanted characters like #, $, and leading/trailing spaces
            f2.write(line1)
            i += 1
    f1.close()
    f2.close()

# Function to backup the original PTX file
def back_ptx():
    f1 = open(ptx_file, 'r')
    f2 = open(back_file, 'w')

    for line in f1.readlines():
        f2.write(line)

    f1.close()
    f2.close()

# Function to read the PTX kernel and generate the temp file for fault injection
# This outputs the number of kernels and their corresponding labels
def read_ptx():
    line_num = 0  # Line number
    f1 = open(back_file, 'r')
    f2 = open(temp_file, 'w')

    for line in f1.readlines():
        line_num += 1
        # Strip leading and trailing whitespaces from the line
        analyze_line = line.strip()
        # If the line is a kernel name, write it to the temp.ptx file
        if analyze_line.startswith('// .globl'):
            f2.write(str(line_num) + " ")  # Write the line number
            kernel_name = analyze_line.split()[-1]  # Get the kernel name
            f2.write(kernel_name + "\n")  # Write the kernel name to the file
        if analyze_line == "":
            continue
        if analyze_line.startswith('//'):
            continue
        if analyze_line.startswith('.'):
            continue
        if analyze_line.startswith('(') or analyze_line.startswith(')'):
            continue
        if analyze_line.startswith('}') or analyze_line.startswith('{'):
            continue
        if analyze_line.startswith('@'):
            continue
        if analyze_line.startswith('ret'):
            continue
        # Skip branch instructions, as they don't have destination registers
        if analyze_line.startswith('bra'):
            continue
        else:
            f2.write(str(line_num))  # Write the line number
            f2.write(" " + analyze_line + "\n")  # Write the instruction line
    f1.close()
    f2.close()

# Main function to execute the process
def main():
    # Step 1: After making keep and dry
    # Step 2: Delete unnecessary lines from dryrun.out
    delete_line()
    print("Create dryrun1.out")
    # Step 3: Backup the PTX file
    back_ptx()
    print("Backup ptx file")
    # Step 4: Generate temp.ptx for fault injection
    read_ptx()
    print("Create temp.txt")

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
