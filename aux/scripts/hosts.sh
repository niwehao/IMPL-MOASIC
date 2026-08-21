# 双机拓扑的唯一事实源(实测依据见 log.md「环境事实」)。
# 沿用 nano-vllm/scripts/hosts.sh 的约定 —— 那套脚本已经踩平了一堆坑。
GPU01_SSH=192.168.100.1          # 直连 100GbE,不要用 hostname(会解析到 2GbE 的 bond0)
GPU01_IP=192.168.100.1
GPU02_IP=192.168.100.2           # 本机 = rank 0..G-1 所在 host = driver
IFNAME=ens5f0np0
REPO=$HOME/CodeRead/xccl/Impl/mosaic-rt
IMPL=$HOME/CodeRead/xccl/Impl        # 四个实现目录的根
