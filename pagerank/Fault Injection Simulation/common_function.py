# -*- coding: utf-8 -*-

# This script is mainly used for:
# 1. Deleting unnecessary lines from dryrun.out
# 2. Backing up the ptx file
# 3. Generating a pt file suitable for fault injection (FI)

ptx_file = "PageRank.ptx"  # The original ptx file
dryrun_before = "dryrun.out"  # The file to be cleaned, containing all compilation commands
dryrun_file = "dryrun1.out"  # The file after removing unnecessary lines, as modification of the ptx file should begin from the ptxas command
back_file = "PageRank1.ptx"  # Backup of the ptx file
temp_file = "temp.ptx"  # Temporary file to store selected lines for fault injection


# Function to delete unnecessary lines in dryrun.out
def delete_line():
    f1 = open(dryrun_before, 'r')
    f2 = open(dryrun_file, 'w')

    i = 1
    for line in f1.readlines():
        # Keep the lines starting from the 20th line, which contains ptxas command
        if i < 20:
            i += 1
            continue
        else:
            line1 = line.strip('#$ ')  # Remove unwanted characters
            f2.write(line1)
            i += 1
    f1.close()
    f2.close()


# Function to back up the ptx file
def back_ptx():
    f1 = open(ptx_file, 'r')
    f2 = open(back_file, 'w')

    for line in f1.readlines():
        f2.write(line)

    f1.close()
    f2.close()


# Function to read a kernel from the ptx file, for selecting instructions suitable for fault injection
# Outputs the number of kernels and the labels of instructions
def read_ptx():
    line_num = 0  # Line number counter
    f1 = open(back_file, 'r')
    f2 = open(temp_file, 'w')

    for line in f1.readlines():
        line_num += 1
        # Strip whitespace from the start and end of each line
        analyze_line = line.strip()
        # If the line is the name of a kernel, write it into temp.ptx
        if analyze_line.startswith('// .globl'):
            f2.write(str(line_num) + " ")  # Write the line number
            kernel_name = analyze_line.split()[-1]  # Extract the kernel name
            f2.write(kernel_name + "\n")  # Write the kernel name at the start of the file
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
        # Branch instructions don't have a destination register, so they are not analyzed
        if analyze_line.startswith('bra'):
            continue
        else:
            f2.write(str(line_num))  # Write the line number
            f2.write(" " + analyze_line + "\n")  # Write the instruction line
    f1.close()
    f2.close()


def main():
    # Step 1: After running make keep and make dry
    # Step 2: Delete unnecessary lines in dryrun.out
    delete_line()
    print("Create dryrun1.out")
    
    # Step 3: Backup the ptx file
    back_ptx()
    print("Backup ptx file")
    
    # Step 4: Generate temp.ptx (only needs to be done once for fault injection)
    read_ptx()
    print("Create temp.txt")


if __name__ == "__main__":
    main()
