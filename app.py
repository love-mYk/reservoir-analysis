"""
储层主控因素分析系统
物以类聚 · 成功被偏爱
by MaYk
"""

import io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import streamlit as st

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 配色 ─────────────────────────────────────────────────────
CAT_COLORS = {
    "重矿物":        "#1565C0",
    "粒度":          "#BF360C",
    "压汞_孔隙结构":  "#1B5E20",
    "岩石学":        "#4A148C",
    "自定义大类A":   "#E65100",
    "自定义大类B":   "#006064",
}
WELL_COLORS = ["#1565C0","#B71C1C","#1B5E20","#4A148C",
               "#E65100","#006064","#880E4F","#37474F"]
DEFAULT_CATS = ["重矿物","粒度","压汞_孔隙结构","岩石学",
                "自定义大类A","自定义大类B"]

# ══════════════════════════════════════════════════════════════
# 分析核心
# ══════════════════════════════════════════════════════════════

def build_dist(df, cols, metric="euclidean"):
    X = StandardScaler().fit_transform(df[cols].values)
    return squareform(pdist(X, metric=metric))

def mantel_test(dX, dY, n_perm=999, method="spearman"):
    n   = dX.shape[0]
    idx = np.tril_indices(n, k=-1)
    vX, vY = dX[idx], dY[idx]
    fn  = spearmanr if method == "spearman" else pearsonr
    r0  = fn(vX, vY)[0]
    pr  = np.array([
        fn(dX[np.ix_(p:=np.random.permutation(n),p)][idx], vY)[0]
        for _ in range(n_perm)
    ])
    return dict(r=r0, p=float(np.mean(np.abs(pr)>=np.abs(r0))), perm_r=pr)

def run_nmds(D, max_iter=500, seed=42):
    m = MDS(n_components=2, metric=False, dissimilarity="precomputed",
            max_iter=max_iter, n_init=10, random_state=seed,
            normalized_stress="auto")
    return m.fit_transform(D), m.stress_

def sig(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"

def fig_to_svg(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def fig_to_png(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()

# ══════════════════════════════════════════════════════════════
# 各图独立绘制（每张返回独立 Figure）
# ══════════════════════════════════════════════════════════════

def fig_mantel_bars(mres):
    """① Mantel Test 汇总条形图"""
    fig, ax = plt.subplots(figsize=(7, max(3, len(mres)*1.1)),
                           facecolor="#F4F6F8")
    cats   = list(mres.keys())
    r_vals = [mres[c]["r"] for c in cats]
    p_vals = [mres[c]["p"] for c in cats]
    colors = [CAT_COLORS.get(c,"#607D8B") for c in cats]

    bars = ax.barh(cats, r_vals, color=colors, height=0.52,
                   edgecolor="white", lw=0.8)
    for bar, r, p in zip(bars, r_vals, p_vals):
        x  = r+0.005 if r>=0 else r-0.005
        ha = "left" if r>=0 else "right"
        ax.text(x, bar.get_y()+bar.get_height()/2,
                f"r={r:.3f}  {sig(p)}",
                va="center", ha=ha, fontsize=10, fontweight="bold")

    best_i = int(np.argmax(r_vals))
    ax.annotate("★ 最强",
                xy=(r_vals[best_i], best_i),
                xytext=(r_vals[best_i]+0.008, best_i+0.35),
                fontsize=9, color="#C62828", fontweight="bold")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Mantel r（Spearman）", fontsize=10)
    ax.set_title("① Mantel Test\n各大类 → 物性  矩阵级整体相关性排行",
                 fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlim(-0.02, max(r_vals)*1.5+0.05)
    fig.tight_layout()
    return fig

def fig_perm_dist(mres):
    """② 置换分布（前2大类）"""
    items = list(mres.items())[:2]
    fig, axes = plt.subplots(1, len(items),
                             figsize=(6*len(items), 4),
                             facecolor="#F4F6F8")
    if len(items)==1:
        axes = [axes]
    for ax, (cat, res) in zip(axes, items):
        c = CAT_COLORS.get(cat,"#607D8B")
        ax.hist(res["perm_r"], bins=40, color=c, alpha=0.55, edgecolor="white")
        ax.axvline(res["r"], color="#C62828", lw=2.2,
                   label=f"观测 r={res['r']:.3f}")
        ax.axvline(-np.abs(res["r"]), color="#C62828", lw=1.2,
                   ls="--", alpha=0.5)
        ax.set_title(f"② 置换检验 — {cat}\n"
                     f"r={res['r']:.3f}  p={res['p']:.4f}  {sig(res['p'])}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Mantel r", fontsize=9)
        ax.set_ylabel("频数",      fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

def fig_heatmap(df_clean, categories, prop_cols):
    """
    ③ 相关热图
    行 = 各大类下的所有变量（按大类顺序排列）
    列 = 物性变量
    左侧色块 = 大类归属
    修复：imshow y轴从上到下(0→n)，inset axes需同向
    """
    # 按大类顺序展开变量
    all_vars = [c for cols in categories.values() for c in cols]
    n_vars   = len(all_vars)
    n_props  = len(prop_cols)

    # Spearman 相关矩阵
    corr_arr = np.array([
        [spearmanr(*df_clean[[v,p]].dropna().values.T)[0]
         if len(df_clean[[v,p]].dropna())>5 else np.nan
         for p in prop_cols]
        for v in all_vars
    ])

    # 图尺寸自适应变量数
    fig_h = max(5, n_vars * 0.45 + 1.5)
    fig_w = max(5, n_props * 1.8 + 3.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#F4F6F8")

    im = ax.imshow(corr_arr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n_props))
    ax.set_xticklabels(prop_cols, fontsize=10, fontweight="bold")
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(all_vars, fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    for i in range(n_vars):
        for j in range(n_props):
            v = corr_arr[i,j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if abs(v)>0.45 else "black")

    # ── 左侧大类色块（关键修复：invert_yaxis 与 imshow 同向）──
    ax_b = ax.inset_axes([-0.28, 0, 0.22, 1])
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(-0.5, n_vars-0.5)
    ax_b.invert_yaxis()          # ← 修复：与 imshow 同向，0在上
    ax_b.axis("off")

    idx = 0
    for cat, cols in categories.items():
        n = len(cols)
        color = CAT_COLORS.get(cat,"#607D8B")
        y_lo  = idx - 0.45
        y_hi  = idx + n - 0.55
        ax_b.fill_betweenx([y_lo, y_hi], 0, 1,
                            color=color, alpha=0.88)
        ax_b.text(0.5, (y_lo+y_hi)/2, cat,
                  ha="center", va="center",
                  fontsize=8, color="white",
                  fontweight="bold", rotation=90)
        idx += n

    # 大类分隔线
    idx = 0
    for cat, cols in categories.items():
        idx += len(cols)
        if idx < n_vars:
            ax.axhline(idx-0.5, color="white", lw=2.5)

    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label("Spearman r", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("③ 相关热图：所有变量 vs 物性\n"
                 "（按大类分组着色 | 变量层面验证主控大类）",
                 fontsize=11, fontweight="bold", pad=22)
    fig.tight_layout()
    return fig

def fig_nmds(coords, labels, stress, title,
             highlight=None, clabel=""):
    """④ NMDS 单张"""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#F4F6F8")
    if highlight is not None:
        sc = ax.scatter(coords[:,0], coords[:,1],
                        c=highlight, cmap="RdYlGn", s=60,
                        alpha=0.88, edgecolors="white", lw=0.4)
        plt.colorbar(sc, ax=ax, label=clabel, shrink=0.85)
    else:
        for i, lab in enumerate(dict.fromkeys(labels)):
            m = labels==lab
            ax.scatter(coords[m,0], coords[m,1],
                       c=WELL_COLORS[i%len(WELL_COLORS)],
                       label=str(lab), s=60, alpha=0.88,
                       edgecolors="white", lw=0.4)
        ax.legend(fontsize=8, title="分组", title_fontsize=8, framealpha=0.7)
    ax.axhline(0, color="lightgray", lw=0.6, ls="--")
    ax.axvline(0, color="lightgray", lw=0.6, ls="--")
    ax.set_xlabel("NMDS1", fontsize=10)
    ax.set_ylabel("NMDS2", fontsize=10)
    ax.set_title(f"{title}\nStress = {stress:.4f}",
                 fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig

def fig_poro_perm(df, poro, perm, garnet, group):
    """⑤ 孔渗交汇图"""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#F4F6F8")
    sub = df[[poro,perm,garnet,group]].dropna()
    mkr = ["o","s","^","D","v","P","*","X"]
    sc  = None
    for i, w in enumerate(sub[group].unique()):
        ws = sub[sub[group]==w]
        sc = ax.scatter(ws[poro],
                        np.log10(ws[perm].clip(lower=1e-6)),
                        c=ws[garnet], cmap="RdYlGn",
                        vmin=sub[garnet].min(), vmax=sub[garnet].max(),
                        marker=mkr[i%len(mkr)], s=65, alpha=0.88,
                        edgecolors="gray", lw=0.3, label=w)
    if sc is not None:
        plt.colorbar(sc, ax=ax, label=f"{garnet} (%)", shrink=0.85)
    ax.legend(fontsize=8, title="分组", title_fontsize=8)
    ax.set_xlabel("孔隙度 (%)", fontsize=10)
    ax.set_ylabel("渗透率  log₁₀(mD)", fontsize=10)
    ax.set_title("⑤ 孔渗交汇图\n同孔不同渗 | 颜色 = 指示变量",
                 fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════
# 模拟数据
# ══════════════════════════════════════════════════════════════
def demo_data(n=40):
    rng = np.random.default_rng(42)
    rows = []
    for wi, w in enumerate(["N1","N2","N3","N4"]):
        g0 = [38,30,18,10][wi]
        for _ in range(n):
            g  = float(np.clip(rng.normal(g0,5),0,70))
            po = float(np.clip(8+0.25*g+rng.normal(0,2),2,25))
            pe = float(max(10**(0.07*g+rng.normal(0,0.7)-1.8),0.01))
            rows.append({
                "well_name":w, "garnet":g,
                "zircon":    float(np.clip(rng.normal(14,4),0,35)),
                "leucoxene": float(np.clip(rng.normal(8,3),0,20)),
                "magnetite": float(np.clip(rng.normal(5,2),0,12)),
                "ilmenite":  float(np.clip(rng.normal(4,2),0,10)),
                "rutile":    float(np.clip(rng.normal(3,1),0,8)),
                "tourmaline":float(np.clip(rng.normal(6,2),0,14)),
                "amphibole": float(np.clip(rng.normal(10,3),0,24)),
                "Mz":        float(1.5+0.015*g+rng.normal(0,0.25)),
                "std_phi":   float(np.clip(rng.normal(0.8,0.2),0.3,2)),
                "Sk":        float(rng.normal(0.05,0.18)),
                "Kg":        float(rng.normal(1.0,0.18)),
                "mean_Gr":   float(np.clip(20+0.4*g+rng.normal(0,5),5,80)),
                "face_porosity":    float(np.clip(po*0.72+rng.normal(0,1),0,20)),
                "pore_diameter":    float(np.clip(50+1.8*g+rng.normal(0,10),5,200)),
                "specific_surface": float(np.clip(rng.normal(5,1),1,10)),
                "throat_radius":    float(np.clip(0.5+0.018*g+rng.normal(0,0.1),0.05,2)),
                "Pc50":             float(np.clip(rng.normal(3,1),0.5,8)),
                "Sd":               float(np.clip(rng.normal(1.5,0.3),0.5,3)),
                "quartz_pct":   float(np.clip(65-0.1*g+rng.normal(0,5),30,90)),
                "feldspar_pct": float(np.clip(rng.normal(12,4),0,30)),
                "lithic_pct":   float(np.clip(rng.normal(8,3),0,20)),
                "cement_pct":   float(np.clip(rng.normal(5,2),0,15)),
                "chlorite_pct": float(np.clip(0.5+0.1*g+rng.normal(0,1),0,10)),
                "porosity":po, "permeability":pe,
            })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="储层主控因素分析系统",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 哲学标题 ──────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 8px 0 4px 0;'>
  <h2 style='color:#1565C0; margin-bottom:2px; font-size:1.7rem;'>
    🪨 储层主控因素分析系统
  </h2>
  <p style='color:#555; font-size:1.05rem; margin:0; letter-spacing:1px;'>
    <b>物以类聚</b> — 是数据分析发现的结构规律
    &nbsp;&nbsp;｜&nbsp;&nbsp;
    <b>成功被偏爱</b> — 是地质研究给出的因果关系
  </p>
</div>
<hr style='margin:8px 0 16px 0;'/>
""", unsafe_allow_html=True)

# ── 侧边栏 ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 分析参数")

    st.subheader("Mantel Test")
    n_perm   = st.slider("置换次数", 99, 9999, 999, 100,
                         help="越多越精确，建议 ≥ 999")
    m_meth   = st.radio("相关方法", ["spearman","pearson"],
                        index=0, horizontal=True)
    metric   = st.radio("距离度量", ["euclidean","cosine"],
                        index=0, horizontal=True)

    st.subheader("NMDS")
    nmds_iter = st.slider("最大迭代次数", 100, 2000, 500, 100)

    st.subheader("导出设置")
    dpi = st.select_slider("PNG 分辨率", [100,150,200,300], value=150)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#888;font-size:0.78rem;'>"
        "by MaYk</div>",
        unsafe_allow_html=True)

# ── 数据导入 ──────────────────────────────────────────────────
st.subheader("📂 数据导入")
col_u, col_prev = st.columns([1, 2])

with col_u:
    use_demo = st.checkbox("使用内置模拟数据（演示）", value=True)
    uploaded = st.file_uploader("上传 CSV / Excel",
                                type=["csv","xlsx","xls"],
                                disabled=use_demo)

if use_demo:
    df = demo_data(40)
    st.info("🔬 内置模拟数据：4口井 × 40样品 × 24指标")
elif uploaded:
    try:
        df = (pd.read_excel(uploaded)
              if uploaded.name.endswith((".xlsx",".xls"))
              else pd.read_csv(uploaded, encoding="utf-8-sig"))
        st.success(f"✅ {df.shape[0]} 行 × {df.shape[1]} 列")
    except Exception as e:
        st.error(f"读取失败：{e}"); st.stop()
else:
    st.warning("请上传数据或勾选模拟数据"); st.stop()

with col_prev:
    st.dataframe(df.head(8), use_container_width=True, height=220)
    c1,c2,c3 = st.columns(3)
    c1.metric("行数", df.shape[0])
    c2.metric("列数", df.shape[1])
    c3.metric("缺失率", f"{df.isnull().mean().mean()*100:.1f}%")

st.divider()

# ── 列名分配 ──────────────────────────────────────────────────
st.subheader("🗂️ 列名分配")
st.caption("将数据列分配到各分析化验大类 — 每大类整体构成一个距离矩阵")

all_cols = list(df.columns)

cg, cp = st.columns(2)
with cg:
    group_col = st.selectbox("🏷️ 分组列（井名/层位）", all_cols, index=0)
with cp:
    default_props = [c for c in ["porosity","permeability"] if c in all_cols] or all_cols[:2]
    prop_cols = st.multiselect("🎯 物性列（目标矩阵）", all_cols,
                                default=default_props)

if len(prop_cols) < 1:
    st.warning("请至少选择 1 个物性列"); st.stop()

n_cats = st.number_input("大类数量", 2, 6, 4, 1)

def guess_defaults(name, cols):
    kw = {
        "重矿物":["garnet","zircon","leucoxene","magnetite",
                  "ilmenite","rutile","tourmaline","amphibole"],
        "粒度":  ["Mz","std","Sk","Kg","Gr","phi"],
        "压汞":  ["face","pore","specific","throat","Pc","Sd"],
        "岩石学":["quartz","feldspar","lithic","cement","chlorite"],
    }
    for k, hints in kw.items():
        if k in name:
            return [c for c in cols
                    if any(h.lower() in c.lower() for h in hints)]
    return []

categories = {}
rows_ui = [st.columns(2) for _ in range((int(n_cats)+1)//2)]

for i in range(int(n_cats)):
    ri, ci = i//2, i%2
    with rows_ui[ri][ci]:
        label = st.text_input(f"大类 {i+1} 名称",
                              value=DEFAULT_CATS[i] if i<len(DEFAULT_CATS) else f"大类{i+1}",
                              key=f"cn_{i}")
        exclude = prop_cols + [group_col]
        avail   = [c for c in all_cols if c not in exclude]
        defs    = [c for c in guess_defaults(label, avail) if c in avail]
        sel = st.multiselect(f"变量列（{label}）", avail,
                             default=defs, key=f"cc_{i}")
        if len(sel)>=2:
            categories[label] = sel
            st.success(f"✅ {len(sel)} 列")
        elif len(sel)==1:
            st.warning("至少需要 2 列")

# 孔渗图设置
with st.expander("🔧 孔渗交汇图设置"):
    cx,cy,cz = st.columns(3)
    poro_col   = cx.selectbox("孔隙度列", prop_cols, index=0)
    perm_col   = cy.selectbox("渗透率列", prop_cols,
                               index=min(1,len(prop_cols)-1))
    non_prop   = [c for c in all_cols if c not in prop_cols+[group_col]]
    garnet_col = cz.selectbox("颜色映射列（石榴石等）", non_prop, index=0)

st.divider()

# ── 运行 ──────────────────────────────────────────────────────
valid_cats = {k:v for k,v in categories.items() if len(v)>=2}
if len(valid_cats) < 2:
    st.warning("请至少配置 2 个有效大类后再运行"); st.stop()

if st.button("🚀 开始分析", type="primary", use_container_width=True):

    # 公共样本
    need     = list({c for v in valid_cats.values() for c in v} | set(prop_cols))
    df_clean = df[need+[group_col]].dropna().reset_index(drop=True)

    if len(df_clean) < 10:
        st.error(f"有效样本仅 {len(df_clean)} 条，无法分析"); st.stop()

    st.info(f"参与分析：**{len(df_clean)}** 样品  |  "
            f"**{len(valid_cats)}** 大类  |  置换 **{n_perm}** 次")

    # Mantel Test
    with st.spinner("⏳ Mantel Test 计算中..."):
        dist_prop = build_dist(df_clean, prop_cols, metric)
        dist_cats = {cat: build_dist(df_clean, cols, metric)
                     for cat, cols in valid_cats.items()}
        mres = {}
        prog = st.progress(0)
        for i,(cat,D) in enumerate(dist_cats.items()):
            mres[cat] = mantel_test(D, dist_prop, n_perm, m_meth)
            prog.progress((i+1)/len(dist_cats), text=f"计算中：{cat}")
        prog.empty()

    mres = dict(sorted(mres.items(), key=lambda x:x[1]["r"], reverse=True))
    best = list(mres.keys())[0]

    # 结果卡片
    st.subheader("📊 Mantel Test 结果")
    mc = st.columns(len(mres))
    for col,(cat,res) in zip(mc, mres.items()):
        col.metric(f"{'★ ' if cat==best else ''}{cat}",
                   f"r = {res['r']:.4f}",
                   f"p={res['p']:.4f} {sig(res['p'])}")
    st.success(f"✅ **【{best}】** 与物性整体相关性最强  "
               f"Mantel r = {mres[best]['r']:.4f}")

    # NMDS
    with st.spinner("⏳ NMDS 降维中..."):
        dist_all   = np.mean(list(dist_cats.values()), axis=0)
        C_best, S_best = run_nmds(dist_cats[best], nmds_iter)
        C_all,  S_all  = run_nmds(dist_all, nmds_iter)

    labels  = df_clean[group_col].values
    g_vals  = df_clean[garnet_col].values if garnet_col in df_clean.columns else None

    # ════════════════════════════════════════════════════════
    # 逐图展示 + 独立 SVG 下载
    # ════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📈 分析图表（每张可独立下载 SVG / PNG）")

    def dl_row(fig, name):
        """在图下方显示 SVG + PNG 两个下载按钮"""
        d1, d2, _ = st.columns([1,1,4])
        d1.download_button(
            f"⬇️ SVG",
            data=fig_to_svg(fig),
            file_name=f"{name}.svg",
            mime="image/svg+xml",
            use_container_width=True,
            key=f"svg_{name}")
        d2.download_button(
            f"⬇️ PNG",
            data=fig_to_png(fig, dpi),
            file_name=f"{name}.png",
            mime="image/png",
            use_container_width=True,
            key=f"png_{name}")

    # ① Mantel 条形
    f1 = fig_mantel_bars(mres)
    st.pyplot(f1, use_container_width=True)
    dl_row(f1, "01_Mantel_条形图")
    plt.close(f1)

    # ② 置换分布
    f2 = fig_perm_dist(mres)
    st.pyplot(f2, use_container_width=True)
    dl_row(f2, "02_置换分布")
    plt.close(f2)

    # ③ 相关热图
    f3 = fig_heatmap(df_clean, valid_cats, prop_cols)
    st.pyplot(f3, use_container_width=True)
    dl_row(f3, "03_相关热图")
    plt.close(f3)

    # ④ NMDS（最强类）
    f4a = fig_nmds(C_best, labels, S_best,
                   f"④ NMDS — {best}（最强大类）",
                   highlight=g_vals,
                   clabel=f"{garnet_col}(%)")
    st.pyplot(f4a, use_container_width=True)
    dl_row(f4a, f"04a_NMDS_{best}")
    plt.close(f4a)

    # ④ NMDS（综合）
    f4b = fig_nmds(C_all, labels, S_all,
                   "④ NMDS — 综合矩阵（所有大类）")
    st.pyplot(f4b, use_container_width=True)
    dl_row(f4b, "04b_NMDS_综合")
    plt.close(f4b)

    # ⑤ 孔渗交汇图
    if all(c in df.columns for c in [poro_col, perm_col, garnet_col, group_col]):
        f5 = fig_poro_perm(df, poro_col, perm_col, garnet_col, group_col)
        st.pyplot(f5, use_container_width=True)
        dl_row(f5, "05_孔渗交汇图")
        plt.close(f5)

    # ── 结果表下载 ──
    st.divider()
    result_df = pd.DataFrame([
        {"大类":cat, "Mantel_r":res["r"], "p值":res["p"],
         "显著性":sig(res["p"]), "是否最强":"★" if cat==best else ""}
        for cat,res in mres.items()
    ])
    st.download_button(
        "⬇️ 下载 Mantel 结果表（CSV）",
        data=result_df.to_csv(index=False, encoding="utf-8-sig"),
        file_name="Mantel_Test结果.csv",
        mime="text/csv",
        use_container_width=True)
