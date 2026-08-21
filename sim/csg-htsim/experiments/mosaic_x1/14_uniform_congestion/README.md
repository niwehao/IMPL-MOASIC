# 14_uniform_congestion — 均匀 A2A:无拥塞逼平,拥塞下谁扛得住(问题 2)

uniform A2A(每对 2MB,无 demand 文件),flex_rail 32 GPU,
taper∈{180(无拥塞),90,60(spine 2×/3× 超额)},nv∈{2400,360};
另一组显式收紧 ECN(`-ecn_threshold_bytes 1048576`)。数据 `results.csv`。

## 主表(completion_us)

| nv | taper | ECMP | FAST | MOSAIC | M vs F |
|---|---|---|---|---|---|
| 2400 | 180 | 7852 | **2345** | 2576 | 0.91× |
| 2400 | 90 | 15605 | **2345** | 2715 | 0.86× |
| 2400 | 60 | 23533 | **2345** | 3164 | 0.74× |
| 360 | 180 | 7852 | **2622** | 2671 | 0.98× |
| 360 | 90 | 15605 | **2622** | 3030 | 0.87× |
| 360 | 60 | 23533 | **2622** | 2864 | 0.92× |
| 360 | 90+ECN | 16567(3360 cuts) | **2622**(0 cuts) | 3206(4793 cuts) | 0.82× |

割下界(与 taper 无关,因跨轨流量可经 NVLink 换轨):B2 = 每 server 跨机
入口 384MB/(8×180G) = **2133µs**。ECMP 的结构性下界(DIRECT 恒过 spine):
每 leaf 上行 168MB/taper = 7466/14933/22400µs——实测 ECMP 贴此界 +4~5%。

## 结论(如实,含对我们不利的部分)

1. **无拥塞:逼平成立**。MOSAIC 距 FAST −1.8%(360)/−9.9%(2400),距
   cut 下界 +21~25%;uniform 是 FAST 主场(全局 permutation 无 incast),
   MOSAIC 付的是 exploration floor 保险费(§6.3 三分法)。
2. **拥塞(taper)只惩罚路径不可变的 ECMP**(2133 的界 vs 23533 实测,
   11×):FAST 的 i→i 同轨结构在 uniform 下从不过 spine,天然免疫;
   MOSAIC 的 DIRECT 探索份额要经 spine,feedback 收敛前付 3~23% 代价,
   但仍 5~7× 好于 ECMP。**"拥塞下反超 FAST"的战场在非均匀负载**
   (见 11_taper:skew 下 FAST 恒 10126、MOSAIC 全 taper 最优)。
3. ECN 收紧不改变排序:FAST 0 cuts(不碰 spine),MOSAIC 4793 cuts
   (exploration 流量被 ECN 压回,+5.8%),ECMP 半盲(cuts 只能全局
   降速,+6.2%)。所有方案共用同一 ECN/pause 底座,公平。

复现:`./sim/tests/htsim_smart_regular -nodes 32 -topology flex_rail
-flex_nv_gbps {nv} -flex_taper_gbps {t} -packet_size 1024 -message_size
2097152 -chunk_size 2097152 -scheduler {s} -workload A2A -all_to_all
-store_forward -quiet [-ecn_threshold_bytes 1048576]`
