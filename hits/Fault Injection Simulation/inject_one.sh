# Fixed paths
golden_file="golden"         # Directory to store the golden output
stdout_file="golden_stdout.txt"
stderr_file="golden_stderr.txt"

result_dir="result" # Directory to store the faulty program outputs

executable_dry="dryrun1.out"  # Compilation command
dry_file="dryrun.out"

# Modified paths
ptx_file="HITS.ptx"   # PTX file
back_file="HITS1.ptx"

source_file="HITS"   # Program source file
executable_file="HITS"  

output_file="HITS.out"   # Program output file
iter_file="iter_info.txt"  # Additional program output
iter_golden_file="golden_iter_info.txt"

# Compile all
make all

# Standard output directory creation
if [ ! -d "$golden_file" ]      
then
    printf "===== Create golden dir =====\n"
    mkdir golden
else
    printf "=====Golden output file already exists=====\n"
fi

# Create golden_stdout.txt and golden_stderr.txt
printf "===== Create golden_stdout.txt and golden_stderr.txt =====\n"
./$executable_file >golden_stdout.txt 2>golden_stderr.txt

# Check if golden_stdout.txt and golden_stderr.txt exist in the current directory
if [ -f "$stdout_file" ] && [ -f "$stderr_file" ]
then
    mv -f golden_stdout.txt golden/golden_stdout.txt
    mv -f golden_stderr.txt golden/golden_stderr.txt
    printf "===== golden_stdout.txt and golden_stderr.txt moved to golden =====\n"
else
    printf "=====golden_stdout.txt or golden_stderr.txt do not exist=====\n"
fi

# Run the application correctly
printf "===== Run application correctly =====\n"

# Check if output file exists in the current directory
if [ -f "$output_file" ]
then
    mv -f HITS.out golden/HITS.out
    printf "===== output file moved to golden =====\n"
else
    printf "===== No output file ====\n"
fi

# Save iteration information
if [ -f "$iter_file" ]
then
    cp -f iter_info.txt golden_iter_info.txt
    printf "===== golden_iter_info.txt has been added =====\n"
else
    printf "=====golden_iter_info.txt does not exist=====\n"
fi

# Clean up some files
make clobber  

# Modify the compile file
# Get PTX file and dryrun.out file
make keep
make dry

# Before Injection
# Remove redundant lines from dryrun.out
python common_function.py
printf "delete_line back_ptx read_ptx"

# Create result directory if it doesn't exist
if [ ! -d "$result_dir" ]
then
    mkdir $result_dir
    printf "\n===== result directory created ====\n"
else
    printf "===== result directory already exists ====\n"
fi

# Set the number of injections
# inject_num=5
middle="_midfile"
DUE_flag=0

# Loop to inject faults
for i in $(seq 1 50000)
do
    printf "**********************************************THE $i INJECTION*****************************************\n"

    # Inject one fault by modifying the PTX file
    python fault-inject.py $i;

    # Create directory for the faulty run results
    mkdir $result_dir/$i
    # Save the modified PTX file as a middle file
    file_name=$i$middle
    cp $ptx_file $result_dir/$i/$file_name

    # If dryrun1.out executable exists
    # Run dryrun1.out and continue with the PTX compilation process
    if test -x $executable_dry ; then
        ./$executable_dry
        OP_MODE=$?
    else
        printf "===== Increase executable permission for dryrun1.out =====\n"
        chmod +x $executable_dry
        ./$executable_dry
        OP_MODE=$?
    fi

    chmod +x $executable_file
    # Create stdout.txt and stderr.txt
    printf "===== Create stdout.txt and stderr.txt =====\n"
    
    # If execution time exceeds 100 seconds, terminate the program and mark it as DUE
    DUE_flag=0
    start_time=$(date +%s)
    ./$executable_file 1>stdout.txt 2>stderr.txt &

    # Get the process ID of the running program
    pid=$!
    
    # Wait for the process to complete, checking every second
    while kill -0 $pid 2>/dev/null; do
        # Calculate the elapsed time
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))
        
        if [ $elapsed_time -ge 20 ]; then
            echo "Execution time exceeded the limit, forcibly exiting"
            DUE_flag=1
            kill -9 $pid  # Force kill the process
            break
        fi
        sleep 1  # Sleep for 1 second
    done

    # Create diff.log to compare outputs
    printf "===== Create diff.log =====\n"
    diff HITS.out golden/HITS.out > diff.log

    # Store the faulty program execution results in the result directory
    cp HITS.out $result_dir/$i/HITS.out
    cp iter_info.txt $result_dir/$i/iter_info.txt
    cp diff.log $result_dir/$i/diff.log

    # After injection
    python parse_diff.py $OP_MODE $i $DUE_flag

done
