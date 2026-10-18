# Beamer 课程

本目录与 Quarto 教材并行维护，用于录制讲课视频。每讲独立编译。

## 第 0 讲：散射理论序言

`lecture00/main.tex`（21 页）是课程序言：先用含时演化建立散射图像，解释时间与空间的乘积结构以及定能分解；再用同一个受迫振动初值问题演示 Picard、Runge–Kutta、Numerov、Fourier、Laplace、Green 函数和极点规定。最后借耦合振子连接分波、径向积分与散射边界条件。

受迫振动用于预演后续方法；正式的散射理论从下一讲开始。

## 编译

需要 latexmk、XeLaTeX、xeCJK、PGFPlots 和 Noto Sans CJK SC 字体。

```bash
cd presentation
make
# presentation/lecture00/build/main.pdf
```

也可以直接运行：

```bash
cd presentation/lecture00
latexmk main.tex
```

`.latexmkrc` 将 PDF 和辅助文件统一输出到 TeX 同级的 `build/`。在 `presentation/` 运行 `make clean` 清理辅助文件并保留 PDF。

## 与教材的联系

- `chapters/01_free_particle.qmd`：自由态
- `chapters/03_lippmann_schwinger.qmd`：LS 方程
- `chapters/05_partial_wave.qmd`：分波
- `chapters/07_analytic_structure.qmd`：极点与解析结构
- `appendix/B_numerical_methods.qmd`：数值方法

幻灯片只给出主线与必要推导；完整证明和数值实现放在教材与示例中。
