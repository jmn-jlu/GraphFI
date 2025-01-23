# Fixed
golden_file="golden"         # Directory to store the golden output
stdout_file="golden_stdout.txt"
stderr_file="golden_stderr.txt"

result_dir="result"          # Directory to store the faulty program output

executable_dry="dryrun1.out" # Compilation command
dry_file="dryrun.out"

# Modified
ptx_file="SCC.ptx"           # PTX file
back_file="SCC1.ptx"

source_file="SCC"            # Program source file
executable_file="SCC"        # Executable file

output_file="SCC.out"        # Program output file
iter_file="iter_info.txt"    # Other program output
iter_golden_file="golden_iter_info.txt" # Golden iteration information

# Compile all
make all
make fi

# Standard output directory
if [ ! -d "$golden_file" ]      
then
    printf "===== Create golden dir =====\n"
    mkdir golden
else
    printf "===== Golden output file already exists =====\n"
fi

# Create golden_stdout.txt and golden_stderr.txt
printf "===== Create golden_stdout.txt and golden_stderr.txt =====\n"
./$executable_file >golden_stdout.txt 2>golden_stderr.txt

# Check if golden_stdout.txt and golden_stderr.txt exist in the current directory
if [ -f "$stdout_file" ]&&[ -f "$stderr_file" ]
then
    mv -f golden_stdout.txt golden/golden_stdout.txt
    mv -f golden_stderr.txt golden/golden_stderr.txt
    printf "===== golden_stdout.txt and golden_stderr.txt moved to golden =====\n"
else
    printf "===== golden_stdout.txt or golden_stderr.txt don't exist =====\n"
fi

# Run correctly
#./$executable_file
printf "===== Run application correctly =====\n"

# If there are any output files in the current directory
if [ -f "$output_file" ]
then
    mv -f SCC.out golden/SCC.out
    printf "===== Output file moved to golden =====\n"
else
    printf "===== No output file found =====\n"
fi

# Added, save iteration information for each round
if [ -f "$iter_file" ]
then
    cp -f iter_info.txt golden_iter_info.txt
    printf "===== golden_iter_info.txt has been added =====\n"
else
    printf "===== golden_iter_info.txt doesn't exist =====\n"
fi

make clobber  # Clean up some files

# Modify the compile file
# Get PTX file and dryrun.out file
make keep
make dry

# Before Injection
# Remove unnecessary lines from dryrun.out
python common_function.py
printf "delete_line back_ptx read_ptx"

# Create result directory if it doesn't exist
if [ ! -d "$result_dir" ]
then
    mkdir $result_dir
    printf "\n===== Result directory created =====\n"
else
    printf "===== Result file already exists =====\n"
fi

# Set times of injection
# inject_num=5
middle="_midfile"
DUE_flag=0
for i in $(seq 1 50000)
do
    printf "********************************************** THE $i INJECTION *****************************************\n"

    # Inject one fault
    # Modify the PTX file for this injection
    python fault-inject.py $i

    # Create fault injection result directory
    mkdir $result_dir/$i
    # Store the intermediate files: PTX file for each injection
    file_name=$i$middle
    cp $ptx_file $result_dir/$i/$file_name

    # If dryrun1.out executable exists
    # Execute dryrun1.out and continue the compilation process with PTXAS
    if test -x $executable_dry ; then
        ./$executable_dry
        OP_MODE=$?
    else
        printf "===== Increase executable of dryrun1.out =====\n"
        chmod +x $executable_dry
        ./$executable_dry
        OP_MODE=$?
    fi
    chmod +x $executable_file
    # Create stdout.txt and stderr.txt
    printf "===== Create stdout.txt and stderr.txt =====\n"

    # If the execution time exceeds time_threshold=100s, terminate the program, mark as DUE
    DUE_flag=0
    start_time=$(date +%s)
    ./$executable_file 1>stdout.txt 2>stderr.txt &
    # Get the process ID of the command
    pid=$!
    
    # Wait for the command to execute, checking every second
    while kill -0 $pid 2>/dev/null; do
        # Calculate the elapsed time
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))
        
        if [ $elapsed_time -ge 20 ]; then
            echo "Program execution time exceeded limit, forced to exit"
            DUE_flag=1
            kill -9 $pid  # Force kill the process
            break
        fi
        sleep 1  # Sleep for 1 second
    done
  
    #printf "===== Create SCC.out =====\n"
    # ./$executable_file

    printf "===== Create diff.log =====\n"

    diff SCC.out golden/SCC.out > diff.log

    # Store the fault injection program execution results in the result directory
    cp SCC.out $result_dir/$i/SCC.out
    cp iter_info.txt $result_dir/$i/iter_info.txt
    cp diff.log $result_dir/$i/diff.log

    # After injection
    python parse_diff.py $OP_MODE $i $DUE_flag

done
