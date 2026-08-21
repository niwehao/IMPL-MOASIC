# NDv2-like no-switch 拓扑状态

## 已完成

- 新增 `topo/ndv2_graph.h`。
- 支持 `TOPOLOGY=ndv2` / `-topology ndv2`。
- 构造 16 个 GPU `SmartNode`：
  - chassis 0: GPU 0-7
  - chassis 1: GPU 8-15
- 不构造 switch node。
- 使用 TE-CCL `ndv2.py` 中 `chassis=2` 的 16x16 matrix 作为来源。
- 将 matrix 中的链路转换为无向 weighted GPU graph：
  - code `23` -> 400 Gbps
  - code `46` -> 200 Gbps
  - code `107` -> 100 Gbps
- htsim 中每条 edge 的 queue service rate 和 sender pacing rate 都使用对应链路速率。
- `tools/teccl_generate_schedule.py` 支持 topo JSON:

```json
{
  "name": "ndv2",
  "node_count": 16
}
```

- TE-CCL 仍通过 `RegularGraph` 读取同一份 generated graph file。
- NDv2-like graph file 标记：
  - `format: htsim-ndv2-like-noswitch-v1`
  - `rate_unit: gbps`
  - `alpha_unit: seconds`
- TE-CCL `RegularGraph` adapter 已支持 `rate_unit=gbps`，会换算成 GB/s 后进入 TE-CCL capacity model。
- 旧 regular graph 不写 `rate_unit`，保持原有行为，避免影响已有 regular 实验。

## 验证结果

基础配置：

```text
Topology: ndv2
NODES: 16
Workload: A2A
chunk_size: 1MiB
num_chunks: 1
```

htsim basic:

```text
validation PASSED
completion_us 4204.48
expected_bytes 251658240
delivered_bytes 251658240
```

TE-CCL schedule:

```text
requested_num_epochs 8
resolved_num_epochs 128
validation PASSED
completion_us 5158.53
expected_bytes 251658240
delivered_bytes 251658240
steps 2127
```

生成的 TE-CCL graph:

```text
node_count 16
edge_count 34
rate_counts:
  400 Gbps: 16
  200 Gbps: 16
  100 Gbps: 2
```

## 未完成 / 近似项

1. 这不是严格 NDv2 复现，只是 no-switch NDv2-like GPU graph。

2. 没有建模 switch node。
   TE-CCL 原始 `ndv2.py` 在其他 chassis 配置下会引入 switch index；当前版本明确忽略 switch，只保留 16 个 GPU。

3. 当前是无向图。
   TE-CCL 原始 matrix 经过 transpose 后可能表达方向性。当前实现把任一方向存在的链路合并成一条双向边。

4. 双向速率不一致时当前取较小非零 capacity code。
   这是保守近似，避免 htsim 和 TE-CCL 在无向图里高估某个方向。

5. htsim 没有使用 alpha。
   graph file 写出了 `0.7us` / `1.3us` alpha，TE-CCL 会读取；但 htsim 当前仍使用统一 pipe delay。

6. 没有建模 MSCCL channel / threadblock / GPU copy engine。
   因此结果只能说明当前 htsim packet-level runtime 下的表现，不能直接等价于 TE-CCL 论文硬件实验。

7. basic routing 仍然按 hop distance 选路，不按链路速率做 shortest-time routing。
   这会让 basic baseline 在 weighted graph 上可能偏弱，也可能导致比较不完全公平。

8. 当前只验证了 A2A、`num_chunks=1`、`chunk_size=1MiB`。
   小消息如 1KB/4KB/16KB，以及 `num_chunks=8` 还没有完成系统比较。

9. TE-CCL 在该 NDv2-like A2A 1MiB 配置下需要 128 epochs 才生成 schedule。
   低 epoch 8/16/32/64 均未生成 schedule。
