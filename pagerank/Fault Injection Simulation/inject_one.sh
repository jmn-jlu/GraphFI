# Fixed directories and files
golden_file="golden"         # Directory for storing the standard output 
stdout_file="golden_stdout.txt"
stderr_file="golden_stderr.txt"

result_dir="result"  # Directory for storing fault injection results

executable_dry="dryrun1.out"  # File containing the compile commands
dry_file="dryrun.out"

# Modified files
ptx_file="PageRank.ptx"   # PTX file
back_file="PageRank1.ptx"

source_file="PageRank"   # Program source file
executable_file="PageRank"  

output_file="PageRank.out"   # Program output file
iter_file="iter_info.txt"  # Program's other output
iter_golden_file="golden_iter_info.txt"

make fi
# Compile all
make all

# Standard output directory
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

# If golden_stdout.txt and golden_stderr.txt exist in the current directory
if [ -f "$stdout_file" ]&&[ -f "$stderr_file" ]
then
    mv -f golden_stdout.txt golden/golden_stdout.txt
    mv -f golden_stderr.txt golden/golden_stderr.txt
    printf "===== golden_stdout.txt and golden_stderr.txt moved to golden =====\n"
else
    printf "===== golden_stdout.txt or golden_stderr.txt don't exist =====\n"
fi

# Run the application correctly
printf "===== Run application correctly =====\n"

# If there are any output files in the current directory
if [ -f "$output_file" ]
then
    mv -f PageRank.out golden/PageRank.out
    printf "===== Output file moved to golden =====\n"
else
    printf "===== No output file found =====\n"
fi

# Add iteration information for each round
if [ -f "$iter_file" ]
then
    cp -f iter_info.txt golden_iter_info.txt
    printf "===== golden_iter_info.txt added =====\n"
else
    printf "===== golden_iter_info.txt doesn't exist =====\n"
fi

make clobber  # Delete unnecessary files

# Modify the compile files
# Get PTX file and dryrun.out file
make keep
make dry

# Before Injection
# Delete unnecessary lines in dryrun.out
python common_function.py
printf "delete_line, back_ptx, read_ptx"

# If the result directory does not exist, create it
if [ ! -d "$result_dir" ]
then
    mkdir $result_dir
    printf "\n===== Result directory created =====\n"
else
    printf "===== Result directory already exists =====\n"
fi

# Set the number of injections
# inject_num=5
middle="_midfile"
DUE_flag=0

for i in $(seq 1 50000)
do
    printf "********************************************** THE $i INJECTION *****************************************\n"

    # Inject one fault
    # Modify PTX file for this injection
    python fault-inject.py $i;

    # Create the result directory for this injection
    mkdir $result_dir/$i
    # Store the intermediate PTX file for this injection
    file_name=$i$middle
    cp $ptx_file $result_dir/$i/$file_name

    # If dryrun1.out is executable, run it
    if test -x $executable_dry ; then
        ./$executable_dry
        OP_MODE=$?
    else
        printf "===== Increase executable permissions of dryrun1.out =====\n"
        chmod +x $executable_dry
        ./$executable_dry
        OP_MODE=$?
    fi

    chmod +x $executable_file
    # Create stdout.txt and stderr.txt
    printf "===== Create stdout.txt and stderr.txt =====\n"
    
    # If the program exceeds the time threshold (100s), terminate it and mark as DUE
    DUE_flag=0
    start_time=$(date +%s)
    ./$executable_file 1>stdout.txt 2>stderr.txt &
    # Get the process ID of the command
    pid=$!
    
    # Wait for the command to finish, checking every second
    while kill -0 $pid 2>/dev/null; do
        # Calculate the elapsed time
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))
        
        if [ $elapsed_time -ge 20 ]; then
            echo "Program execution time exceeded the limit, forcibly exiting"
            DUE_flag=1
            kill -9 $pid  # Forcefully kill the process
            break
        fi
        sleep 1  # Sleep for 1 second
    done
  
    # Create diff.log
    printf "===== Create diff.log =====\n"
    diff PageRank.out golden/PageRank.out > diff.log

    # Store the fault injection results in the result directory
    #cp PageRank.out $result_dir/$i/PageRank.out
    #cp iter_info.txt $result_dir/$i/iter_info.txt
    #p diff.log $result_dir/$i/diff.log

    # After injection, parse the diff
    python parse_diff.py $OP_MODE $i $DUE_flag

done
