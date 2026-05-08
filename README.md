# 散射理论 · 中文教学仓库

本仓库是一本"边读边算"的散射理论中文教科书。每一章既讲清楚理论的核心结构，也给出可以直接运行的 Python 代码与可视化。

定位：

- **独立教学站**：用 [Quarto](https://quarto.org/) 构建，本地预览即可。
- **理论代码并进**：每个 `.qmd` 自包含理论摘要、numpy/scipy 实现、可视化与数值校验。
- **中文优先**。

## 目录结构

- `chapters/` —— 核心理论 8 章（v0.1-alpha 仅含第 1 章）
- `examples/` —— 5 个具体可解势能模型
- `_quarto.yml`、`_macros.qmd`、`index.qmd` —— 站点配置与首页

## 本地构建

1. 安装 Quarto CLI：参见 https://quarto.org/docs/get-started/
2. 准备 Python 环境：

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

3. 启动本地预览（默认浏览器自动打开 `http://localhost:xxxx`）：

```bash
quarto preview
```

或一次性构建静态站点：

```bash
quarto render
# 输出在 _site/
```

## 与配套资料的关系

- 主线 LaTeX 教材：`~/workspace/nutstore/physics/ScatteringTheory/new_zhiguang_slides/note/`（理论详证）
- 博客笔记：`tianbaiting.github.io/docs/sci/minimal_theory/scatteringTheory/`（个别专题深入）

本仓库定位为"可计算化的精简书"，不重复主线教材的全部推导。

## License

代码部分采用 MIT；文字部分采用 CC-BY-4.0。
