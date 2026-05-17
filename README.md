# 储层主控因素分析系统

> 物以类聚 — 是数据分析发现的结构规律  
> 成功被偏爱 — 是地质研究给出的因果关系

**by MaYk**

---

## 功能

| 模块 | 说明 |
|------|------|
| Mantel Test | 各大类分析化验矩阵 vs 物性矩阵的整体相关性 |
| 相关热图 | 变量层面验证主控大类，按大类分组着色 |
| NMDS | 非度量多维度分析，可视化"物以类聚"的空间格局 |
| 孔渗交汇图 | 同孔不同渗现象展示 |
| SVG/PNG 导出 | 每张图独立下载 |

---

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器自动打开 http://localhost:8501

---

## 打包为 Windows EXE

### 方法一：GitHub Actions（推荐）

1. 上传本项目到 GitHub 仓库
2. Actions 自动触发，在 `windows-latest` 上打包
3. 下载 Artifact：`储层主控因素分析系统-Windows`

手动触发：仓库页面 → Actions → Build Windows EXE → Run workflow

发布版本：打一个 tag 自动生成 Release 并附 EXE：
```bash
git tag v1.0.0
git push origin v1.0.0
```

### 方法二：本地打包

```bash
pip install pyinstaller
python -m streamlit info   # 获取 streamlit 路径
pyinstaller launcher.py --onefile --noconsole --name "储层主控因素分析系统" --add-data "app.py;."
```

EXE 在 `dist/` 目录下，双击运行即可。

---

## 数据格式

CSV 或 Excel，每行一个样品，每列一个分析化验指标：

| well_name | garnet | zircon | … | porosity | permeability |
|-----------|--------|--------|---|----------|--------------|
| N1 | 35.2 | 12.1 | … | 8.5 | 0.45 |

列名中英文均可，在界面中手动分配到各大类。

---

## 项目结构

```
├── app.py                        # Streamlit 主程序
├── launcher.py                   # EXE 启动入口
├── requirements.txt              # 依赖
├── README.md
└── .github/
    └── workflows/
        └── build.yml             # GitHub Actions 打包流程
```
