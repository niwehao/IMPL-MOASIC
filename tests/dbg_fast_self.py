import os, sys
sys.path.insert(0, "/home/weihaoni/CodeRead/xccl/Impl/mosaic-rt/python")
import mosaic_py as mp

rank = int(sys.argv[1])
os.environ["MOSAIC_JOB"] = "911"
os.environ["MOSAIC_LOG"] = "3"
c = mp.Comm(rank=rank, nranks=2, gpus_per_host=1, hosts=os.environ.get("MOSAIC_HOSTS","127.0.0.1"),
            sched="fast", use_gpu=1)
n = 2
# 带 self 流量:scounts = [2048, 2048](rank0),模拟 torch 测试形态
sc = [2048, 2048]
sd = [0, 2048]
rc = [2048, 2048]
rd = [0, 2048]
import ctypes
send = (ctypes.c_uint8 * 4096)(*([rank * 16 + (i % 7) for i in range(4096)]))
recv = (ctypes.c_uint8 * 4096)()
c.alltoallv(ctypes.addressof(send), sc, sd, ctypes.addressof(recv), rc, rd)
print(f"[r{rank}] a2a done", flush=True)
c.close()
