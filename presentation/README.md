# Beamer 课程

本目录与 Quarto 教材并行维护，用于录制讲课视频。每讲独立编译。

## 第一讲：散射理论概述

`lecture01/main.tex`（21 页）介绍散射问题的脉络、含时演化与定态描述的关系，再借受迫振子介绍直接积分、傅里叶与 Laplace 变换、Green 函数、极点规定。最后连接到径向 Schrödinger 方程、渐近匹配、动量空间 LS 方程与矩阵求逆。

它是导论，不是将受迫振动当作散射理论的主题，也不试图在第一讲完整推导所有技术。

## 编译

需要 XeLaTeX、xeCJK 和 Noto Sans CJK SC 字体。

```bash
cd presentation
make
# presentation/lecture01/main.pdf
```

也可以直接运行：

```bash
cd presentation/lecture01
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

## 与教材的联系

- `chapters/01_free_particle.qmd`：自由态
- `chapters/03_lippmann_schwinger.qmd`：LS 方程
- `chapters/05_partial_wave.qmd`：分波
- `chapters/07_analytic_structure.qmd`：极点与解析结构
- `appendix/B_numerical_methods.qmd`：数值方法

幻灯片只给出主线与必要推导；完整证明和数值实现放在教材与示例中。
