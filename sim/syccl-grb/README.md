# SyCCL: Exploiting Symmetry for Efficient Collective Communication Scheduling

SyCCL is a scalable collective schedule synthesizer that aims to quickly synthesize near-optimal schedules for production-scale machine-learning jobs. It leverages collective and topology symmetries to decompose the original collective communication demand into smaller sub-demands within smaller topology subsets. Specifically, SyCCL proposes efficient search strategies to quickly explore potential sub-demands, synthesizes corresponding sub-schedules, and integrates these sub-schedules into complete schedules. For more details, please refer to our paper in [SIGCOMM 2025](https://dl.acm.org/doi/10.1145/3718958.3750499).

SyCCL formulates the schedule problems of sub-demands as Mixed Integer Linear Programming (MILP) problems for *AllGather/ReduceScatter* and Linear Programming (LP) problems for *AllToAll*. While an internal solver was used for the SyCCL paper, for the convenience of other researchers, we adopt the non-commercial solver SCIP in this repository.

Note that the efficiency of solving linear programming problems may be limited by SCIP, so using faster solvers (e.g. Gurobi) is recommended for your own research if available.

This repository replaces the SCIP interface used in MILP modeling with the gurobi interface. You would need to additionally set the environment variable ```GUROBI_HOME``` during the build process, and setup a [Gurobi license](https://www.gurobi.com/downloads/) via the environment variable ```GRB_LICENSE_FILE``` when running the code.

## Building Synthesizer

1. Install prerequisites
```bash
apt-get install wget cmake g++ m4 xz-utils libgmp-dev unzip zlib1g-dev libboost-program-options-dev libboost-serialization-dev libboost-regex-dev libboost-iostreams-dev libtbb-dev libreadline-dev pkg-config git liblapack-dev libgsl-dev flex bison libcliquer-dev gfortran file dpkg-dev libopenblas-dev rpm
```

2. Download the [SCIP](https://www.scipopt.org) optimization suite scipoptsuite-x.y.z.tar and install it

```bash
tar xvf scipoptsuite-x.y.z.tar
cd scipoptsuite-x.y.z
mkdir build
cd build
cmake .. -DAUTOBUILD=on -DTPI=tny
make -j
make check
make install
```

The default installation path is ```/usr/local```. You can modify the path by changing the command:

```bash
cmake .. -DAUTOBUILD=on -DTPI=tny -DCMAKE_INSTALL_PREFIX=/path/to/SCIP
```

3. Install SCIPpp interface for C++
```bash
git clone https://github.com/scipopt/SCIPpp.git
cd SCIPpp
cmake . -DCMAKE_PREFIX_PATH=/path/to/SCIP # change the path of SCIP, e.g., /usr/local
make ScipPP
make install
```
The default installation path is also ```/usr/local```.

4. Build syntheiszer
``` bash
git clone https://github.com/aliyun/symccl.git
cd symccl
mkdir build
cd build
cmake .. -DSCIP_SUITE_DIR=/path/to/SCIP -DSCIP_PP_DIR=/path/to/SCIPpp
make -j
```

If it succeeds, a binary ```synthesize``` will be generated in the directory ```build```.

## Running Synthesizer

SyCCL uses configuration file in json format as the input. We provide several configuration files in directory ```config``` as examples, and you can modify these files to adapt to your topology. In addition, we provide a script ```gen_single_config.py``` in ```scripts``` as an example to generate new configuration files. In the ```build``` directory, run the following command, which use ```default.json``` by default:

``` bash
./synthesize solve
```

To specify the configuration file, add the ```-f``` parameter:
``` bash
./synthesize -f ../config/a100-8gpu-4nic-clos-ag.json solve
```

The result will be stored in a json file indicated by the ```solve_output``` field in the configuration file. 

To run the test cases in our paper, we provide a script ```runexp.sh``` in ```scripts``` to generate the configuration files and execute the synthesizer with the generated files.

## Real-world Evaluation

To evaluate the performance of SyCCL in real world, we exploit [MSCCL](https://github.com/Azure/msccl) as the runtime, which uses an xml file to describe the scheduling strategy.
To transfer the result file to MSCCL xml file, we provide a script ```transfer_to_msccl.py``` in ```scripts```:

```bash
git clone https://github.com/microsoft/msccl-tools.git
cd msccl-tools
pip install -r requirements.txt
pip install .
python3 transfer_to_msccl.py <result json file> | <result json directory>
```

Then, follow the guidance of [MSCCL](https://github.com/Azure/msccl) to execute the perftest or end-to-end training. Note that for Megatron, the communication of data parallelism (DP) is in-place and that of tensor parallelism (TP) is out-of-place. Thus, ```transfer_to_msccl.py``` generates two types of xml files indicated by ```inplace=True``` and ```inplace=False``` respectively.

## Extending Synthesizer

### Customizing sketches

SyCCL exploits sketches to reduce the the searching space and break down the collective communication into multiple sub-demands. To explore more potential sketches, SyCCL provides an interface to support customized sketches in ```src/Sketch/SeachECSet.cpp```. You can define your sketches in the function ```Algorithm::CustomizeSketch()``` and set ```"customize_sketch": true``` in the configuration file. As the sketches searched by SyCCL do not depend on the type of collective communication, you can set ```"save_sketch": true``` to save the sketches in the file indicated by ```sketch_path``` and set ```"use_sketch_input": true``` to re-use the saved sketches.

### Exploring alternative solvers

Additionally, SyCCL provides the interface for user to explore alternative solving methods(e.g. Greedy/Heuristic based) other than MILP/LP based methods for better scalability. Refer to the base class in ```include/Solver/AlgoSolver.h``` for more details.

### Migrating to other topologies

Although the modeling works for any input topology, the code currently only works with with certain switched topologies. For direct connect topologies or more complex switched topologies, you may need to modify the code accordingly. 

Comments inside this codebase describes how the code makes assumptions about the topology and how to modify it for other topologies.

## License
SyCCL is an open source project developed by Alibaba Cloud and licensed under the Apache License (Version 2.0)
This product contains various third-party components under other open source licenses. 
See the NOTICE file for more information.