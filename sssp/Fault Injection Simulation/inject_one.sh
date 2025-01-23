# Fixed configuration
golden_file="golden"         # Directory for storing standard output files
stdout_file="golden_stdout.txt"
stderr_file="golden_stderr.txt"

result_dir="result"          # Directory for storing output after fault injection

dry_file="dryrun.out"
executable_dry="dryrun1.out" # Compiled PTX executable file

# Configuration based on the source program
source_file="SSSP"
executable_file="SSSP"       # Executable file 
output_file="SSSP.out"       # Program output file
ptx_file="SSSP.ptx"
back_file="SSSP1.ptx"        # Backup of the PTX file

iter_file="iter_info.txt"    # File containing iteration information
iter_golden_file="golden_iter_info.txt" # File containing total iteration count and fault injection status

make fi                      # Compile the fault injection
# Compile all the files
make all

# Create the directory to store golden output if not already present
if [ ! -d "$golden_file" ]      
then
    printf "===== Create golden dir =====\n"
    mkdir golden
else
    printf "===== Golden output files already exist =====\n"
fi

# Create golden_stdout.txt and golden_stderr.txt
printf "===== Create golden_stdout.txt and golden_stderr.txt =====\n"
./$executable_file >golden_stdout.txt 2>golden_stderr.txt

# If the golden stdout and stderr files exist, move them to the golden directory
if [ -f "$stdout_file" ]&&[ -f "$stderr_file" ]
then
    mv -f golden_stdout.txt golden/golden_stdout.txt
    mv -f golden_stderr.txt golden/golden_stderr.txt
    printf "===== Moved golden_stdout.txt and golden_stderr.txt to golden directory =====\n"
else
    printf "===== golden_stdout.txt or golden_stderr.txt do not exist =====\n"
fi

# Run the application correctly
printf "===== Run application correctly =====\n"

# If the output file exists, move it to the golden directory
if [ -f "$output_file" ]
then
    mv -f SSSP.out golden/SSSP.out
    printf "===== Moved output file to golden directory =====\n"
else
    printf "===== No output file found ====\n"
fi

# Save iteration information if the file exists
if [ -f "$iter_file" ]
then
    cp -f iter_info.txt golden_iter_info.txt
    printf "===== Added golden_iter_info.txt =====\n"
else
    printf "===== golden_iter_info.txt does not exist =====\n"
fi

make clobber  # Clean up unnecessary files

# Modify the compile file to get the PTX file and dryrun.out file
make keep
make dry

# Before Fault Injection
# Remove unnecessary lines from dryrun.out
python common_function.py
printf "delete_line back_ptx read_ptx"

# Create result directory if not exists
if [ ! -d "$result_dir" ]
then
    mkdir $result_dir
    printf "\n===== Result directory created ====\n"
else
    printf "===== Result directory already exists ====\n"
fi

# Set the number of injections
middle="_midfile"
DUE_flag=0
for i in $(seq 1 50000)
do
    printf "********************************************** Injection $i *****************************************\n"

    # Modify PTX file for this injection
    python fault-inject.py $i;

    # Create the directory for this fault injection result
    mkdir $result_dir/$i
    # Store the intermediate PTX file for this injection
    file_name=$i$middle
    cp $ptx_file $result_dir/$i/$file_name

    # If dryrun1.out is executable, run it
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
    # Create stdout.txt and stderr.txt for fault injection execution
    printf "===== Create stdout.txt and stderr.txt =====\n"
    
    # If the program execution time exceeds the threshold (100 seconds), terminate and mark as DUE
    DUE_flag=0
    start_time=$(date +%s)
    ./$executable_file 1>stdout.txt 2>stderr.txt &
    # Get the process ID of the running command
    pid=$!
    
    # Wait for the command to finish, checking every second
    while kill -0 $pid 2>/dev/null; do
        # Calculate elapsed time
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))
        
        # If the time exceeds the threshold, kill the process and mark as DUE
        if [ $elapsed_time -ge 100 ]; then
            echo "Execution time exceeded the limit, forcefully terminating"
            DUE_flag=1
            kill -9 $pid  # Force kill the process
            break
        fi
        sleep 1  # Sleep for 1 second
    done
  
    # Create diff.log to compare outputs
    printf "===== Create diff.log =====\n"
    diff SSSP.out golden/SSSP.out > diff.log

    # Store the output of this fault injection execution in the result directory
    cp SSSP.out $result_dir/$i/SSSP.out
    cp iter_info.txt $result_dir/$i/iter_info.txt
    cp diff.log $result_dir/$i/diff.log

    # After Fault Injection
    python parse_diff.py $OP_MODE $i $DUE_flag

done
