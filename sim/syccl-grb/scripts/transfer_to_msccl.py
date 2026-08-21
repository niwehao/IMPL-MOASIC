# Copyright (c) 2025, Alibaba Cloud and its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import math
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import *
import functools
from functools import partial
import os
import itertools
import multiprocessing as mp
from contextlib import redirect_stdout
from io import StringIO


def convert_chunk_allgather(chunk_gpu, chunk_index, chunk_num, min_chunk_size, chunk_sizes):
    chunk_cnt = chunk_sizes[(chunk_gpu, chunk_index)] // min_chunk_size
    new_chunk_index = chunk_gpu * chunk_num + sum(chunk_sizes[(chunk_gpu, i)] // min_chunk_size for i in range(chunk_index))
    return new_chunk_index, chunk_cnt

def convert_chunk_alltoall(chunk_gpu, chunk_index, chunk_num, min_chunk_size, chunk_sizes, gpu_num):
    chunk_cnt = chunk_sizes[(chunk_gpu, chunk_index)] // min_chunk_size
    receive_gpu, chunk_index = chunk_index % gpu_num, chunk_index // gpu_num
    new_chunk_index = receive_gpu * chunk_num + sum(chunk_sizes[(chunk_gpu, i * gpu_num + receive_gpu)] // min_chunk_size for i in range(chunk_index))
    return new_chunk_index, chunk_cnt

def do_lower(full_chunk_size, coll, gpu_num, algo_schedule, size, protocol, fusion, depnop, instance, inplace):
    print(f"=====> Lowering {coll} with {gpu_num} GPUs, full_chunk_size={full_chunk_size}, size={size}, protocol={protocol}, fusion={fusion}, depnop={depnop}, instance={instance}, inplace={inplace}")
    chunk_sizes = {tuple(c[0]): c[1] for c in algo_schedule["Schedule"]["chunk_sizes"]}
    # min_chunk_size = math.gcd(*(chunk_sizes.values()))
    chunk_size_list = list(chunk_sizes.values()) 
    min_chunk_size = functools.reduce(math.gcd, chunk_size_list)
    chunk_num = full_chunk_size // min_chunk_size
    if chunk_num > 32:
        raise ValueError(f"chunk_num {chunk_num} is too large, skipping this algorithm")
    if coll == "alltoall":
        for rank in range(gpu_num): chunk_sizes[(rank, rank * chunk_num)] = full_chunk_size

    slice_send_time = {}
    for e in algo_schedule["SliceSentTime"]:
        ((chunk_gpu, chunk_index), src_gpu, dst_gpu), send_time_list = e
        if coll == "allgather" or coll == "reducescatter":
            new_chunk_index, chunk_cnt = convert_chunk_allgather(chunk_gpu, chunk_index, chunk_num, min_chunk_size, chunk_sizes)
        elif coll == "alltoall":
            new_chunk_index, chunk_cnt = convert_chunk_alltoall(chunk_gpu, chunk_index, chunk_num, min_chunk_size, chunk_sizes, gpu_num)
        slice_send_time[((chunk_gpu, new_chunk_index, chunk_cnt), src_gpu, dst_gpu)] = send_time_list

    slice_recv_time = {}
    for e in algo_schedule["SliceReceivedTime"]:
        ((chunk_gpu, chunk_index), src_gpu, dst_gpu), recv_time_list = e
        if coll == "allgather" or coll == "reducescatter":
            new_chunk_index, chunk_cnt = convert_chunk_allgather(chunk_gpu, chunk_index, chunk_num, min_chunk_size, chunk_sizes)
        elif coll == "alltoall":
            new_chunk_index, chunk_cnt = convert_chunk_alltoall(chunk_gpu, chunk_index, chunk_num, min_chunk_size, chunk_sizes, gpu_num)
        slice_recv_time[((chunk_gpu, new_chunk_index, chunk_cnt), src_gpu, dst_gpu)] = recv_time_list

    topology = fully_connected(gpu_num)
    pre_sorted_op = sorted(slice_send_time.keys(), key=lambda e:(slice_send_time[e][0], (slice_recv_time[e][0], slice_recv_time[e][-1])))
    if coll == "allgather":
        collective = AllGather(gpu_num, chunk_num, inplace)
    elif coll == "alltoall":
        collective = AllToAll(gpu_num, chunk_num, False)
    elif coll == "reducescatter":
        collective = ReduceScatter(gpu_num, chunk_num, inplace)
        pre_sorted_op = [(chunk, dst, src) for (chunk, src, dst) in pre_sorted_op]
        pre_sorted_op.reverse()

    chunk_copy_on_gpu = defaultdict(lambda: {})
    with MSCCLProgram(f"syn_{coll}_{gpu_num}gpu_{size}", topology, collective, instance, 
        protocol=protocol, threadblock_policy=ThreadblockPolicy.auto, instr_fusion=fusion, dependence_nop=depnop):
        if coll == "alltoall":
            for rank in range(gpu_num): chunk(rank, Buffer.input, rank * chunk_num, chunk_num).copy(rank, Buffer.output, rank * chunk_num)
        elif not inplace and coll == "allgather":
            for rank in range(gpu_num): chunk(rank, Buffer.input, 0, chunk_num).copy(rank, Buffer.output, rank * chunk_num)
        for (src_chunk, src_gpu, dst_gpu) in pre_sorted_op:
            src_chunk_gpu, src_chunk_index, src_chunk_cnt = src_chunk
            if src_chunk_gpu == src_gpu or coll == "reducescatter":
                if coll == "allgather":
                    if inplace: c = chunk(src_chunk_gpu, Buffer.output, src_chunk_index, src_chunk_cnt)
                    else: c = chunk(src_chunk_gpu, Buffer.input, src_chunk_index - src_chunk_gpu * chunk_num, src_chunk_cnt)
                elif coll == "alltoall":
                    c = chunk(src_chunk_gpu, Buffer.input, src_chunk_index, src_chunk_cnt)
                elif coll == "reducescatter":
                    c = chunk(src_gpu, Buffer.input, src_chunk_index, src_chunk_cnt)
            else:
                assert src_chunk in chunk_copy_on_gpu[src_gpu], f"chunk{src_chunk} is not on gpu{src_gpu} when it tries to send"
                c = chunk_copy_on_gpu[src_gpu][src_chunk]
            if coll == "allgather":
                c = c.copy(dst_gpu, Buffer.output, src_chunk_index)
                chunk_copy_on_gpu[dst_gpu][src_chunk] = c
            elif coll == "alltoall":
                if src_chunk_index // chunk_num == dst_gpu:
                    c.copy(dst_gpu, Buffer.output, src_chunk_gpu * chunk_num + (src_chunk_index % chunk_num))
                else:
                    c = c.copy(dst_gpu, src_chunk)
                    chunk_copy_on_gpu[dst_gpu][src_chunk] = c
            elif coll == "reducescatter":
                c_dst = chunk(dst_gpu, Buffer.input, src_chunk_index, src_chunk_cnt)
                c_dst.reduce(c)
        if not inplace and coll == "reducescatter":
            for rank in range(gpu_num):
                chunk(rank, Buffer.input, rank * chunk_num, chunk_num).copy(rank, Buffer.output, 0)
        output = StringIO()
        with redirect_stdout(output):
            XML()
            xml = output.getvalue()
        assert Check(), "collective result incorrect"
    return xml

def parse_synthesizer_json(file):
    assert(file.endswith('.json'))
    with open(file) as f:
        r = f.read()
        j = json.loads(r)

    gpu_num = j["ngpus"]
    coll = j["coll_name"]
    full_chunk_size = j["chunk_size_byte"]
    size = full_chunk_size * gpu_num
    algorithms = j["algorithms"]
    return [gpu_num, coll, full_chunk_size, size, algorithms]

PROTOCOLS = ["Simple", "LL", "LL128"]
INSTANCES = [1, 2, 4]
INPLACES = [True, False]

def gen_single_xml(algo_path: str, xml_dir: str):
    try:
        [gpu_num, coll, full_chunk_size, size, algorithms] = parse_synthesizer_json(algo_path)
    except Exception as e:
        print(f"=====> !!! not valid JSON")
        return
    
    base_algo_name = os.path.basename(algo_path).replace('.json', '')
    for algo_id, algo in enumerate(algorithms):
        for (proto, fusion, depnop, instance, inplace) in itertools.product(PROTOCOLS, [False], [False], INSTANCES, INPLACES):
            try:
                algo_name = f"{base_algo_name}-{algo_id}-{proto}-fuse={fusion}-depnop={depnop}-i={instance}-inplace={inplace}"
                print(f"=====> {algo_name}")
                xml = do_lower(full_chunk_size, coll, gpu_num, algo["final_schedule"], size, proto, fusion, depnop, instance, inplace)
                xml_file = f"{algo_name}.xml"
                xml_path = os.path.join(xml_dir, xml_file)
                with open(xml_path, "w+") as f:
                    print(f"=====> {xml_path}")
                    f.write(xml)
                if coll == "allgather":
                    rs_xml = do_lower(full_chunk_size, "reducescatter", gpu_num, algo["final_schedule"], size, proto, fusion, depnop, instance, inplace)
                    xml_file = f"{algo_name}-reducescatter.xml"
                    xml_path = os.path.join(xml_dir, xml_file)
                    with open(xml_path, "w+") as f:
                        print(f"=====> {xml_path}")
                        f.write(rs_xml)
            except Exception as e:
                print(f"Error: {e}")

def gen_all_xmls(algo_dir: str, xml_dir: str):
    # Collect all JSON file paths first
    algo_paths = []
    for algo_dir, dirs, algos in os.walk(algo_dir):
        for algo_file in algos:
            if algo_file.endswith('.json'):
                algo_paths.append(os.path.join(algo_dir, algo_file))

    # Create a partial function with fixed xml_dir
    gen_single_xml_partial = partial(gen_single_xml, xml_dir=xml_dir)

    # Use all available CPU cores (you can adjust this number if needed)
    num_processes = 96

    # Create a process pool and map the work
    with mp.Pool(processes=num_processes) as pool:
        # Execute the processing in parallel
        pool.map(gen_single_xml_partial, algo_paths)

if __name__ == "__main__":
    file_path = sys.argv[1]
    if os.path.isfile(file_path):
        xml_dir = os.path.join(os.path.dirname(file_path), 'xml')
        if not os.path.exists(xml_dir):
            os.makedirs(xml_dir)
        gen_single_xml(file_path, xml_dir)
    elif os.path.isdir(file_path):
        xml_dir = os.path.join(file_path, 'xml')
        if not os.path.exists(xml_dir):
            os.makedirs(xml_dir)
        gen_all_xmls(file_path, xml_dir)
    else:
        print(f"Not valid file or directory: {file_path}")
