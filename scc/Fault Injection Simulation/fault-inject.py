# -*- coding: utf-8 -*-


import os, io, sys, re, math, random
import numpy as np


"""
-- Fault Injection (FI)
   FI - V1.0 - July 31, 2023
   Nan Jiang
   College of Computer Science and Technology, Jilin University
"""


app_name = "SCC"
ptx_file = "SCC.ptx"
dryrun_before = "dryrun.out"
# compile from ptxas
dryrun_file = "dryrun1.out"
back_file = "SCC1.ptx"
temp_file = "temp.ptx"
basic_file = "outcome.txt"
iter_file = "golden_iter_info.txt"


"""
- Specify or obtain instruction list for FI
- Output: Instruction list for FI
"""
def instruction_list():
    ins_list = []

    line_number = 0
    f = open(back_file,'r')

    for line in f.readlines():
        
        line_number += 1
        analyze_line = line.strip()
        ana_ins = analyze_line.split(' ')

        if analyze_line.startswith('//') or analyze_line.startswith('.') or analyze_line.startswith(')') \
                or analyze_line.startswith('{') or analyze_line.startswith('}') or analyze_line == '' \
                or analyze_line.startswith('$L__BB') or analyze_line.startswith('ret') or analyze_line.startswith('_'):
            continue
        # analyze_line.startswith(bra)
        elif analyze_line.startswith('@') or analyze_line.startswith('bra') or analyze_line[0].split('.')[-1] == 'f32':
            continue
        elif analyze_line.startswith('st'):
            continue
        else:
            ins_list.append(line_number)
            continue
            
    # print(ins_list)
    return ins_list

"""
- Specify or randomly choose target instruction for FI
- Input: Instruction list
- Output: Line number of target instruction
"""
def inject_line_num(ins_list):
    if ins_list:
        target_line = random.choice(ins_list)
        return target_line
    else:
        print("ins_list is empty")



"""
- Analyze target instruction
- Input: Line number of target instruction
- Output: Instruction opcode, destination register digit (1/32/64), destination register type, destination register (%r11)

"""
def analyze_ins(target_line):
    f = open(temp_file, 'r')
    for line in f.readlines():
        split_line = line.split() 
        if len(split_line) > 2:

            if split_line[0] == str(target_line):
                ins_str = split_line[1]
                des_str = split_line[2]

                ins_str_list = ins_str.split('.') 

                if len(ins_str_list) == 2:
                    ins_opcode = ins_str_list[0] # mov,mad,ld....
                    reg_str = ins_str_list[-1]  # s32,u32,s64,b32,f32
                    if reg_str == "pred":
                        reg_digit = "1"
                        reg_type = "pred"
                    else:
                        reg_digit = re.sub("[^0-9]", "", reg_str)
                        reg_type = re.sub("[^a-z]", "", reg_str)
                # ld param u64
                else:
                    ins_opcode = ins_str_list[0]  # mov,mad,ld....
                    reg_str = ins_str_list[-1]  # s32,u32,s64,b32,f32
                    if ins_opcode == "setp":
                        reg_digit = "1"
                        reg_type = "pred"
                    elif ins_opcode == "mul" or ins_opcode == "mad":
                        if ins_str_list[1] == "wide":
                            reg_digit = str(int(re.sub("[^0-9]", "", reg_str)) * 2)
                        else:
                            reg_digit = re.sub("[^0-9]", "", reg_str)
                        reg_type = re.sub("[^a-z]", "", reg_str)
                    elif ins_opcode == "atom":
                        ins_opcode == ins_str_list[0] + '.' + ins_str_list[1]
                        reg_digit = re.sub("[^0-9]", "", reg_str)
                        reg_type = re.sub("[^a-z]", "", reg_str) 
                    elif ins_opcode == "cvt":
                        reg_digit = re.sub("[^0-9]", "", ins_str_list[-2])
                        reg_type = ins_str_list[-2]
                    elif des_str.startswith('%rs'):
                        reg_digit = 16
                        reg_type = re.sub("[^a-z]", "", reg_str) 
                    elif des_str.startswith('%rd'):
                        reg_digit = 64
                        reg_type = re.sub("[^a-z]", "", reg_str) 
                    else:
                        reg_digit = re.sub("[^0-9]", "", reg_str)
                        reg_type = re.sub("[^a-z]", "", reg_str)


                des_reg = des_str.split(',')[0]
                return ins_opcode, reg_digit, reg_type, des_reg, reg_str



"""
- Generate random fault value
- Input: Register bits of the destination register of target instruction
- Output: Fault value in hexadecimal, random bit
"""
def random_bit(reg_digit):
    # for pred register
    if reg_digit == 1:
        ran_bit = random.randint(0, 1)
    else:
        ran_bit = random.randint(0, int(reg_digit)-1)  # generate random value
    dec_num = str(int(math.pow(2, ran_bit))).replace(".0", "")  # float to str
    fault_value = hex(np.compat.long(str(dec_num), 10))  # convert to hexadecimal
    fault_value = str(fault_value).strip('L')
    return fault_value, ran_bit



"""
- Specify or randomly select thread for FI
- Input: Thread id range
- Output: Thread id for FI
"""
def random_thread2(thread_x, thread_y):
    random_x = random.randint(0, thread_x - 1)
    random_y = random.randint(0, thread_y - 1)
    return random_x, random_y

def random_thread1(thread_x):
    random_x = random.randint(0, thread_x - 1)
    return random_x


def random_loop_time():
    total_loop = 50   
    ran_loop = random.randint(1,total_loop)
    return ran_loop
    
def random_iter_time(iter_time):
    ran_iter = random.randint(1,iter_time)
    return ran_iter

"""
- Judge whether the target instruction is in a loop
- Input: Line number of the target instruction
- Output: Y/N
"""

def in_loop(target_line):
    if 94 < target_line < 154:
        return '1'
    elif 161 < target_line < 181:
        return '1'
    elif 201 < target_line < 222:
        return '1'
    elif 231 < target_line < 289:  # gather
        return '1'
    elif 431 < target_line < 457:
        return '1'
    elif 472 < target_line < 498:
        return '1'        
    else:
        return '0'


def get_loop_reg(target_line):
    loop_reg = 'null'
    if 94 < target_line < 154:
        loop_reg = "%r89"
    elif 161 < target_line < 181:
        loop_reg = "%r98"
    elif 201 < target_line < 222:
        loop_reg = "%r103"
    elif 231 < target_line < 289:  # gather
        loop_reg = "%r105"
    elif 431 < target_line < 457:
        loop_reg = "%r32"    
    elif 472 < target_line < 498:
        loop_reg = "%r35"     
    else:
        loop_reg = 'null'
    return loop_reg


def get_kernel_name_and_param(target_line):
    param1 = ''
    param2 = ''
    kernel_name = 'null'
    if  15 < target_line < 297 :
        kernel_name = '_Z10scc_gatherPiS_PKiS1_S1_S1_PbS2_iiS_iS_i'  	
        param1 = '_Z10scc_gatherPiS_PKiS1_S1_S1_PbS2_iiS_iS_i_param_12'
        param2 = '_Z10scc_gatherPiS_PKiS1_S1_S1_PbS2_iiS_iS_i_param_13'
    if  299 < target_line < 361 :
        kernel_name = '_Z9scc_applyPiPKiS_S1_iS_i'
        param1 = '_Z9scc_applyPiPKiS_S1_iS_i_param_5'
        param2 = '_Z9scc_applyPiPKiS_S1_iS_i_param_6'
    if  363 < target_line < 501 :
        kernel_name = '_Z11scc_scatterPiPKiS1_S1_S1_PbS2_iiS_S_iS_i'
        param1 = '_Z11scc_scatterPiPKiS1_S1_S1_PbS2_iiS_S_iS_i_param_12'
        param2 = '_Z11scc_scatterPiPKiS1_S1_S1_PbS2_iiS_S_iS_i_param_13'
    return kernel_name, param1, param2

    
"""
- Fault inject function: Inject one fault
- Input: Line number of target instruction, thread dimension, random thread id x, random thread id y,
         thread x register, thread y register, loop register, destination register type, fault value, destination register digit,
         destination register, label number, isntruction type, register str, last param, random loop depth for FI
- Output: generate modified PTX file for FI

inject_one_fault(target_line, thread_num, target_x, com_thread_x, loop_reg, reg_type, fault_value, reg_digit, des_reg,
                     instruction_type,reg_str,ran_loop,ran_iter)
"""

def inject_one_fault(target_line,
                     thread_num,
                     target_x, 
                     com_thread_x, 
                     loop_reg,
                     reg_type,
                     fault_value,
                     reg_digit,
                     des_reg,
                     instruction_type,
                     reg_str,
                     ran_loop,
                     ran_iter,param1, param2
                     ):
    line_num = 0
    pred_reg_num = 0
    rd_reg_num = 0
    r_reg_num = 0

    pfile = open(ptx_file, "w")
    bfile = open(back_file, 'r')


    for line in bfile.readlines():
        line_num += 1

        if in_loop(target_line) == '0':
            # thread 1
            if thread_num == 1:
                if line.strip().startswith('.reg'):
                    type_item = line.strip().split()[1]  

                    if type_item == ".pred":
                        pred_reg_num = re.sub("[^0-9]", "", line.strip().split()[2])  
                        after_pred = int(pred_reg_num) + 3  
                        str_1 = ".reg .pred   %p<" + str(after_pred) + ">;"
                        pfile.write('    ' + str_1 + '\n')  

                    elif type_item == '.b64':
                        rd_reg_num = re.sub("[^0-9]", "", line.strip().split()[2])  
                        after_rd = int(rd_reg_num) + 1
                        str_1 = ".reg .b64   %rd<" + str(after_rd) + ">;"
                        pfile.write('    ' + str_1 + '\n')  

                    elif type_item == '.b32':
                        r_reg_num = re.sub("[^0-9]", "", line.strip().split()[2]) 
                        after_r = int(r_reg_num) + 1
                        str_1 = ".reg .b32   %r<" + str(after_r) + ">;"
                        pfile.write('    ' + str_1 + '\n')  
                    else:
                        pfile.write(line)  

                # target line

                elif line_num == int(target_line):
                    pfile.write(line)
                    
                    if reg_type == "pred":
                        insert_str1 = "ld.param.u64    %rd" + str(rd_reg_num) + ", [" + param1 + "];"
                        insert_str2 = "ld.param.u32    %r" + str(int(r_reg_num)) + ", [" + param2 + "];"

                        insert_str3 = "setp.eq." + str(instruction_type) + " " + "%p" + str(pred_reg_num) \
                                      + ", " + str(com_thread_x) + "," + str(target_x) + ";"

                        insert_str4 = "setp.eq." + str(instruction_type) + " " + "%p" + str(int(pred_reg_num) + 1) \
                                     + ", %r" + str(int(r_reg_num)) + ", " + str(ran_iter) + ";"
                        insert_str5 = "and.pred    " + "%p" + str(int(pred_reg_num) + 1) + ", %p" + str(int(pred_reg_num) + 1) + ", %p" + str(pred_reg_num) + ";"
                        insert_str6 = "@!%p" + str(int(pred_reg_num) + 1) + " bra    $L__BB1_100;"

                        insert_str7 = "st.global.s32    [%rd" + str(rd_reg_num) + "], 1;"
                        insert_str8 = "xor.pred     " + str(des_reg) + ", " + str(des_reg) + ", 0x1;"
                        insert_str9 = "$L__BB1_100:"

                        pfile.write('       ' + insert_str1 + '\n')
                        pfile.write('       ' + insert_str2 + '\n')
                        pfile.write('       ' + insert_str3 + '\n')
                        pfile.write('       ' + insert_str4 + '\n')
                        pfile.write('       ' + insert_str5 + '\n')
                        pfile.write('       ' + insert_str6 + '\n')
                        pfile.write('       ' + insert_str7 + '\n')
                        pfile.write('       ' + insert_str8 + '\n')
                        pfile.write(insert_str9 + '\n')
                        pass

                    else:
                        insert_str1 = "ld.param.u64    %rd" + str(rd_reg_num) + ", [" + param1 + "];"
                        insert_str2 = "ld.param.u32    %r" + str(int(r_reg_num)) + ", [" + param2 + "];"

                        insert_str3 = "setp.eq." + str(instruction_type) + " " + "%p" + str(pred_reg_num) \
                                      + ", " + str(com_thread_x) + "," + str(target_x) + ";"
                        insert_str4 = "setp.eq." + str(instruction_type) + " " + "%p" + str(int(pred_reg_num) + 1) \
                                     + ", %r" + str(int(r_reg_num)) + ", " + str(ran_iter) + ";"
                        insert_str5 = "and.pred    " + "%p" + str(int(pred_reg_num) + 1) + ", %p" + str(int(pred_reg_num) + 1) + ", %p" + str(pred_reg_num) + ";"
                        insert_str6 = "@!%p" + str(int(pred_reg_num) + 1) + " bra    $L__BB1_100;"

                        insert_str7 = "st.global.s32    [%rd" + str(rd_reg_num) + "], 1;"
                        insert_str8 = "xor.b" + str(reg_digit) + "      " + str(des_reg) + ", " + str(des_reg) + ", " + str(fault_value) + ";"
                        insert_str9 = "$L__BB1_100:"

                        pfile.write('       ' + insert_str1 + '\n')
                        pfile.write('       ' + insert_str2 + '\n')
                        pfile.write('       ' + insert_str3 + '\n')
                        pfile.write('       ' + insert_str4 + '\n')
                        pfile.write('       ' + insert_str5 + '\n')
                        pfile.write('       ' + insert_str6 + '\n')
                        pfile.write('       ' + insert_str7 + '\n')
                        pfile.write('       ' + insert_str8 + '\n')
                        pfile.write(insert_str9 + '\n')

                else:
                    pfile.write(line)


        else:
            if thread_num == 1:
                if line.strip().startswith('.reg'):
                    type_item = line.strip().split()[1]  
                    if type_item == ".pred":
                        pred_reg_num = re.sub("[^0-9]", "", line.strip().split()[2])  
                        after_pred = int(pred_reg_num) + 4
                        str_1 = ".reg .pred   %p<" + str(after_pred) + ">;"
                        pfile.write('    ' + str_1 + '\n')  
                    elif type_item == '.b64':
                        rd_reg_num = re.sub("[^0-9]", "", line.strip().split()[2])  
                        after_rd = int(rd_reg_num) + 1
                        str_1 = ".reg .b64   %rd<" + str(after_rd) + ">;"
                        pfile.write('    ' + str_1 + '\n')  
                    elif type_item == '.b32':
                        r_reg_num = re.sub("[^0-9]", "", line.strip().split()[2]) 
                        after_r = int(r_reg_num) + 1
                        str_1 = ".reg .b32   %r<" + str(after_r) + ">;"
                        pfile.write('    ' + str_1 + '\n')  
                    else:
                        pfile.write(line)  
                elif line_num == int(target_line):
                    pfile.write(line)
                    if reg_type == "pred":
                        insert_str1 = "ld.param.u64    %rd" + str(rd_reg_num) + ", [" + param1 + "];"
                        insert_str2 = "ld.param.u32    %r" + str(int(r_reg_num)) + ", [" + param2 + "];"
                        insert_str3 = "setp.eq." + str(instruction_type) + " " + "%p" + str(pred_reg_num) \
                                      + ", " + str(com_thread_x) + "," + str(target_x) + ";"
                        insert_str4 = "setp.eq." + str(instruction_type) + " " + "%p" + str(int(pred_reg_num) + 1) \
                                     + ", %r" + str(int(r_reg_num)) + ", " + str(ran_iter) + ";"
                        insert_str5 = "setp.eq." + str(instruction_type) + " " + "%p" + str(int(pred_reg_num) + 2) \
                                     + ", " + str(loop_reg) + ", " + str(ran_loop) + ";"
                        insert_str6 = "and.pred    " + "%p" + str(int(pred_reg_num) + 1) + ", %p" + str(int(pred_reg_num) + 1) + ", %p" + str(pred_reg_num) + ";"
                        insert_str7 = "and.pred    " + "%p" + str(int(pred_reg_num) + 2) + ", %p" + str(int(pred_reg_num) + 2) + ", %p" + str(int(pred_reg_num) + 1) + ";"
                        insert_str8 = "@!%p" + str(int(pred_reg_num) + 2) + " bra    $L__BB1_100;"
                        insert_str9 = "st.global.s32    [%rd" + str(rd_reg_num) + "], 1;"
                        insert_str10 = "xor.pred     " + str(des_reg) + ", " + str(des_reg) + ", 0x1;"
                        insert_str11 = "$L__BB1_100:"

                        pfile.write('       ' + insert_str1 + '\n')
                        pfile.write('       ' + insert_str2 + '\n')
                        pfile.write('       ' + insert_str3 + '\n')
                        pfile.write('       ' + insert_str4 + '\n')
                        pfile.write('       ' + insert_str5 + '\n')
                        pfile.write('       ' + insert_str6 + '\n')
                        pfile.write('       ' + insert_str7 + '\n')
                        pfile.write('       ' + insert_str8 + '\n')
                        pfile.write('       ' + insert_str9 + '\n')
                        pfile.write('       ' + insert_str10 + '\n')
                        pfile.write(insert_str11 + '\n')
                        pass
                    else:
                        insert_str1 = "ld.param.u64    %rd" + str(rd_reg_num) + ", [" + param1 + "];"
                        insert_str2 = "ld.param.u32    %r" + str(int(r_reg_num)) + ", [" + param2 + "];"
                        insert_str3 = "setp.eq." + str(instruction_type) + " " + "%p" + str(pred_reg_num) \
                                      + ", " + str(com_thread_x) + "," + str(target_x) + ";"
                        insert_str4 = "setp.eq." + str(instruction_type) + " " + "%p" + str(int(pred_reg_num) + 1) \
                                     + ", %r" + str(int(r_reg_num)) + ", " + str(ran_iter) + ";"
                        insert_str5 = "setp.eq." + str(instruction_type) + " " + "%p" + str(int(pred_reg_num) + 2) \
                                     + ", " + str(loop_reg) + ", " + str(ran_loop) + ";"
                        insert_str6 = "and.pred    " + "%p" + str(int(pred_reg_num) + 1) + ", %p" + str(int(pred_reg_num) + 1) + ", %p" + str(pred_reg_num) + ";"
                        insert_str7 = "and.pred    " + "%p" + str(int(pred_reg_num) + 2) + ", %p" + str(int(pred_reg_num) + 2) + ", %p" + str(int(pred_reg_num) + 1) + ";"

                        insert_str8 = "@!%p" + str(int(pred_reg_num) + 2) + " bra    $L__BB1_100;"
                        insert_str9 = "st.global.s32    [%rd" + str(rd_reg_num) + "], 1;"
                        insert_str10 = "xor.b" + str(reg_digit) + "      " + str(des_reg) + ", " + str(des_reg) + ", " + str(fault_value) + ";"
                        insert_str11 = "$L__BB1_100:"

                        pfile.write('       ' + insert_str1 + '\n')
                        pfile.write('       ' + insert_str2 + '\n')
                        pfile.write('       ' + insert_str3 + '\n')
                        pfile.write('       ' + insert_str4 + '\n')
                        pfile.write('       ' + insert_str5 + '\n')
                        pfile.write('       ' + insert_str6 + '\n')
                        pfile.write('       ' + insert_str7 + '\n')
                        pfile.write('       ' + insert_str8 + '\n')
                        pfile.write('       ' + insert_str9 + '\n')
                        pfile.write('       ' + insert_str10 + '\n')
                        pfile.write(insert_str11 + '\n')
                else:
                    pfile.write(line) 

    pfile.close()
    bfile.close()


def get_iter_sum(file_path):    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if lines:
                line = lines[-2].strip()  # Get the second to last line
                values = line.split(':')
                return int(values[0])
            else:
                print("The file is empty")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except ValueError:
        print("The second to last line is not an integer")


def random_thread(file_path,iter):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):
                line1 = lines[i].strip()
                line2 = lines[i+1].strip()
                values1 = line1.split(':')
                if len(values1) == 2:
                    first_num, second_num = int(values1[0]), int(values1[1]) 
                    if first_num == iter:
                        values2=line2.split(',')
                        id = random.randint(0,second_num-1) 
                        return second_num,int(values2[id])
            print(f"No match found for {iter}")
    except FileNotFoundError:
        print("File not found2")




def main():
    itera_time = int(sys.argv[1])
    iter_time = get_iter_sum(iter_file)  # Get the total number of iterations from the iteration info file
    ran_iter = random_iter_time(iter_time)  # Randomly select an iteration number to inject a fault

    thread_x, target_x = random_thread(iter_file, ran_iter)  # Get the number of active threads/vertices for the selected iteration
    print(f'Fault-injected iteration: {ran_iter}, active thread count in this iteration: {thread_x}')
    target_x = target_x - 1
    print(f'Fault-injected thread ID: {target_x}, fault-injected node: {target_x + 1}')

    thread_num = 1  # One-dimensional thread block
    instruction_type = "s32"

    ins_list = instruction_list()  # Get the list of instructions eligible for fault injection
    target_line = inject_line_num(ins_list)  # Get the target instruction line for fault injection
    print(f'Fault-injected instruction line: {target_line}')

    kernel_name, param1, param2 = get_kernel_name_and_param(target_line)  # Get the kernel name and its parameters
    print(f'Fault-injected kernel function: {kernel_name}')

    # Analyze the target instruction
    ins_opcode, reg_digit, reg_type, des_reg, reg_str = analyze_ins(target_line)
    print(f'Fault-injected instruction details: {ins_opcode}, {reg_digit}, {reg_type}, {des_reg}, {reg_str}')

    # Fault value
    fault_value, bit = random_bit(reg_digit)
    print(f'Flipped value: {fault_value}, flipped bit: {bit}')

    # Check if it's in a loop and the loop depth
    loop = in_loop(target_line)
    print(f'Is it in a loop: {loop}')

    if loop == '1':
        # In a loop, randomly select a loop iteration for fault injection
        ran_loop = random_loop_time()
        print(f'Selected loop: {ran_loop}')
    else:
        ran_loop = '0'

    # Compare thread ID
    com_thread_x = '%r1'
    # Loop register
    loop_reg = get_loop_reg(target_line)

    inject_one_fault(target_line, thread_num, target_x, com_thread_x, loop_reg, reg_type, fault_value, reg_digit, des_reg,
                    instruction_type, reg_str, ran_loop, ran_iter, param1, param2)

    # Write the results to a file
    with open(basic_file, 'a') as f:
        f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}".format(
            itera_time, app_name, kernel_name, ran_iter, target_x, target_x + 1,
            bit, fault_value, target_line, ins_opcode, reg_digit, reg_type, des_reg, ran_loop
        ))



if __name__ == "__main__":
    main()
