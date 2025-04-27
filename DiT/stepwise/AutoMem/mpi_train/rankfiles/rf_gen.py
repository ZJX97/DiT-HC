#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import math

def generate_rankfile(total_processes, ip_list, cores_per_process):
    """
    
    total_processes
    ip_list ["29.204.25.53", "29.204.25.76"]
    cores_per_process 
    
    """
    
    if not ip_list:
        raise ValueError("IP list cannot be empty")
    
    cpu_per_node = 2  
    cores_per_cpu = 304  
    cores_per_node = cpu_per_node * cores_per_cpu 
    
    total_cores_needed = total_processes * cores_per_process
    
    processes_per_node = cores_per_node // cores_per_process
    
    nodes_needed = math.ceil(total_processes / processes_per_node)
    
    if nodes_needed > len(ip_list):
        cycles_needed = math.ceil(nodes_needed / len(ip_list))
    else:
        cycles_needed = 1
        ip_list = ip_list[:nodes_needed]
    
    rankfile_content = []
    rank = 0
    
    while rank < total_processes:
        for ip in ip_list:
            for cpu_id in range(cpu_per_node):
                cores_available = cores_per_cpu
                procs_on_this_cpu = cores_available // cores_per_process
                
                for i in range(procs_on_this_cpu):
                    if rank >= total_processes:
                        break
                    
                    start_core = i * cores_per_process
                    #end_core = start_core + cores_per_process - 1
                    end_core = start_core + cores_per_process 

                    
                    if end_core >= cores_per_cpu:
                        end_core = cores_per_cpu - 1
                    
                    start_core = start_core + 4

                    rank_entry = f"rank {rank:4d} = {ip} slot={cpu_id}:{start_core}"
                    rankfile_content.append(rank_entry)
                    
                    rank += 1
                    
                    if rank >= total_processes:
                        break
                
                if rank >= total_processes:
                    break
            
            if rank >= total_processes:
                break
    
    return "\n".join(rankfile_content)

def parse_input():
    """
    
    (total_processes, ip_list, cores_per_process) 
    """
    if len(sys.argv) != 4:
        print("usage reference: python generate_rankfile.py <process_num> <iplist> <cores_per_process>")
        print("IP list format: [ip1,ip2,ip3,...]")
        sys.exit(1)
    
    total_processes = int(sys.argv[1])
    
    #parse ip list
    ip_input = sys.argv[2]
    ip_match = re.match(r'\[(.*)\]', ip_input)
    if not ip_match:
        print("error: ip list format are supposed to be: [ip1,ip2,...]")
        sys.exit(1)
    
    ip_list = ip_match.group(1).split(',')
    ip_list = [ip.strip() for ip in ip_list if ip.strip()]  # 去除空白项
    
    cores_per_process = int(sys.argv[3])
    
    return total_processes, ip_list, cores_per_process

def main():
    try:
        total_processes, ip_list, cores_per_process = parse_input()
        rankfile = generate_rankfile(total_processes, ip_list, cores_per_process)
        print(rankfile)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
