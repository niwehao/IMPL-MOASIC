# 37 组:switch-centric(两层 Clos)incast 及其他现象

**日期**:2026-08-01。**报告**:`Note/explore/2026-0801-x37-switch-centric.md`。
**契约**:n=32,clos_leafs=4,host 100G,leaf 上行默认 2:1
(`-clos_uplink_gbps` 覆盖);per-src 预算 T̄=65,011,712B 与主线均匀契约
逐字节相同(`tools/gen_incast.py`,确定性);LB=`tools/clos_bounds.py`
(含 (s+d)%S 钉链 cut),全部手算核对。

## 批次与文件

- `results.csv`:整理表(`collect.py` 从原始 log 生成;列 = side/run/
  status/completion_us/lb_us/x_lb/queue_drops/notes);`plot.py`→`plot.png`。
- 原始:`cache/nonuniform/x37_eps.log`(P1 incast 扫 + P1b buffer
  补偿 + P2 oversub{800,400,200,100}×{uniform,zipf} + P3 spines=4
  collision + P4 mixed;`script/x37_eps.sh`);探针
  `cache/nonuniform/x37_probe_{buf,wave}.txt`(incast1 min-buffer 二分
  128-2048MB;wave 定律 W∈{12,24})。

## 关键结论(细节与数学在报告)

1. **incast 差异化在 buffer 轴**:completion 全员贴 ingress 割界
   (≤0.3%);m=1 时 MOSAIC 默认 1GB 零丢包,FAST/ECMP 打爆(53.8 万
   丢包)需 2GB。min-buffer:MOSAIC = **C/W**(W=6/12/24 → 阈值
   (256,512]/(128,256]/(64,128]MB,三连减半命中模型),blast ≈ 0.95C
   ∈(1024,2048];W 加倍的 completion 代价 ≤0.11%。
2. **PC 纯 incast 中性**(单 sink 串行化,序无关;逐位同 base),但
   oversub×zipf 下 PC 四点精确贴界 40791.3,均匀下把 MOSAIC 从
   −7~11% 拉到 −1~3%(ECMP 是均匀主场,全线 1.00×)。
3. **collision 负边界**:spine 钉死后四方案都贴钉链割界(collide/
   balanced 恰 4.00×),时间调度救不了 routing collision。

## 复现

```bash
script/x37_eps.sh    # + 探针命令见报告 §5
python3 experiments/mosaic_x1/37_switch_centric/collect.py
/home/weihaoni/miniconda3/envs/myenv/bin/python3 experiments/mosaic_x1/37_switch_centric/plot.py  # 需 matplotlib
```
