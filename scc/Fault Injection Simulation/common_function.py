# -*- coding: utf-8 -*-

# Backup PTX files and generate PTX files that can be used for fault injection
ptx_file = "SCC.ptx"  # Original PTX file
dryrun_before = "dryrun.out"  # File that needs extra content removed, contains all compilation commands
dryrun_file = "dryrun1.out"  # File after removing extra content, required to execute from the PTXAS command after modifying the PTX file
back_file = "SCC1.ptx"  # Backup of the original PTX file
temp_file = "temp.ptx"  # Temporary PTX file


# Remove extra lines from dryrun.out
def delete_line():
    f1 = open(dryrun_before, 'r')
    f2 = open(dryrun_file, 'w')

    i = 1
    for line in f1.readlines():
        # Keep lines from the 20th line in dryrun.out, starting from the PTXAS section, remove lines above
        if i < 20:
            i += 1
            continue
        else:
            line1 = line.strip('#$ ')
            f2.write(line1)
            i += 1
    f1.close()
    f2.close()


# Backup the PTX file
def back_ptx():
    f1 = open(ptx_file, 'r')
    f2 = open(back_file, 'w')

    for line in f1.readlines():
        f2.write(line)

    f1.close()
    f2.close()


# Read a kernel's PTX to identify instructions for fault injection
# Output the number of kernels and labels
def read_ptx():
    line_num = 0  # Line number
    f1 = open(back_file, 'r')
    f2 = open(temp_file, 'w')

    for line in f1.readlines():
        line_num += 1
        # Remove leading and trailing spaces from each line
        analyze_line = line.strip()
        # Check if the line is a kernel name and write it to temp.ptx
        if analyze_line.startswith('// .globl'):
            f2.write(str(line_num) + " ")  # Write line number
            kernel_name = analyze_line.split()[-1]  # Kernel name
            f2.write(kernel_name + "\n")  # Kernel name at the top of the file
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
        # Skip branch instructions as they have no destination registers
        if analyze_line.startswith('bra'):
            continue
        else:
            f2.write(str(line_num))  # Write line number
            f2.write(" " + analyze_line + "\n")  # Write instruction line
    f1.close()
    f2.close()


def main():
    # 1 After running make keep and make dry
    # 2 Remove extra lines from dryrun.out once
    delete_line()
    print("Create dryrun1.out")
    # 3 Backup the PTX file once
    back_ptx()
    print("Backup PTX file")
    # 4 Generate temp.ptx, performed only once for multiple fault injections
    read_ptx()
    print("Create temp.txt")


if __name__ == "__main__":
    main()
