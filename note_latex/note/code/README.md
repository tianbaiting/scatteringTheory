# 数值实验代码

每个脚本对应书中一个章节的"数值实验"小节。运行后会向 `../figures/` 输出对应的 PDF 图。

## 依赖

```bash
pip install -r requirements.txt
```

## 运行

```bash
cd code/
python3 ch07_pole_bypass.py        # 生成 ../figures/ch07_pole_bypass.pdf 等
```

或一次性重生成全部：

```bash
for f in ch*.py; do python3 "$f"; done
```

## 索引

| 脚本 | 主题 | 对应章节 |
|---|---|---|
| `ch00_wavepacket.py` | 高斯波包自由演化 | 量子力学回顾 |
| `ch04_polarization.py` | 极化散射观测量 | 含自旋两粒子散射 |
| `ch06_wignerD.py` | Wigner D 函数与螺度振幅 | 自旋与极化 |
| `ch07_pole_bypass.py` | Green 函数极点绕走（4 种方法） | Green 算符 |
| `ch08_born_convergence.py` | Born 级数收敛 | Born 级数 |
| `ch09_lippmann_schwinger.py` | LS 方程数值解 | 定态散射 |
| `ch10_numerov_phase.py` | Numerov 法相移与散射长度 | 分波分析 |
| `ch11_jost_pole_levinson.py` | S 矩阵极点与 Levinson 验证 | 解析性质 |
| `ch12_coulomb.py` | Coulomb 波函数 + Rutherford | Coulomb 散射 |
| `ch14_resonance_breitwigner.py` | Breit-Wigner + 时间延迟 + Khalfin 尾 | 共振 |

所有脚本只用 numpy / scipy / matplotlib / mpmath，无外部依赖。
