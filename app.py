"""
CompMoE-CFI Web Tool — UHPC Strength Prediction & Sustainable Mix Optimization
Streamlit app with actual 5-model production ensemble.

Run: streamlit run app.py
"""
import os, sys, json, pickle
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Model architecture (minimal, matching compmoe_cfi.py) ──────
class SubsystemEncoder(nn.Module):
    def __init__(self, d, e, dr=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, e*2), nn.LayerNorm(e*2), nn.GELU(), nn.Dropout(dr),
            nn.Linear(e*2, e), nn.LayerNorm(e), nn.GELU())
    def forward(self, x): return self.net(x)

class CAG(nn.Module):
    def __init__(self, dims, e, h, K, dr=0.1):
        super().__init__()
        self.enc = nn.ModuleList([SubsystemEncoder(d, e, dr) for d in dims])
        self.att = nn.MultiheadAttention(e, h, dropout=dr, batch_first=True)
        self.norm = nn.LayerNorm(e)
        n = len(dims)
        self.head = nn.Sequential(nn.Linear(e*n, e), nn.GELU(), nn.Dropout(dr), nn.Linear(e, K))
        self.temp = nn.Parameter(torch.ones(1))
    def forward(self, subs):
        embs = [enc(x) for enc, x in zip(self.enc, subs)]
        stack = torch.stack(embs, dim=1)
        a, _ = self.att(stack, stack, stack)
        a = self.norm(a + stack)
        flat = a.reshape(a.size(0), -1)
        logits = self.head(flat)
        t = self.temp.abs().clamp(min=0.5, max=5.0)
        return F.softmax(logits/t, dim=-1)

class LinearExpert(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Linear(d, 1)
    def forward(self, x): return self.net(x).squeeze(-1)

class ShallowExpert(nn.Module):
    def __init__(self, d, h=512, dr=0.15):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dr), nn.Linear(h, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class ResidualExpert(nn.Module):
    def __init__(self, d, h1=256, h2=128, dr=0.15):
        super().__init__()
        self.input_proj = nn.Linear(d, h1); self.bn0 = nn.BatchNorm1d(h1)
        self.block = nn.Sequential(nn.Linear(h1, h1), nn.BatchNorm1d(h1), nn.GELU(), nn.Dropout(dr),
                                     nn.Linear(h1, h2), nn.BatchNorm1d(h2))
        self.proj = nn.Linear(h1, h2)
        self.head = nn.Sequential(nn.GELU(), nn.Dropout(max(0.02, dr*0.5)), nn.Linear(h2, 1))
    def forward(self, x):
        x = F.gelu(self.bn0(self.input_proj(x)))
        x = F.gelu(self.block(x) + self.proj(x))
        return self.head(x).squeeze(-1)

class PhysicsResidualPath(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 1); self.alpha = nn.Parameter(torch.tensor(0.1))
    def forward(self, x, moe):
        a = torch.sigmoid(self.alpha)
        return a * self.linear(x).squeeze(-1) + (1 - a) * moe

def build_diverse_experts(d, K, h1, h2, dr):
    experts = nn.ModuleList()
    for i in range(K):
        if i == 0: experts.append(LinearExpert(d))
        elif i <= 2: experts.append(ShallowExpert(d, h1*2, dr))
        else: experts.append(ResidualExpert(d, h1, h2, dr))
    return experts

class CompMoE_CFI(nn.Module):
    def __init__(self, input_dim, subsystem_dims, num_experts=5, embed_dim=64, num_heads=1,
                 expert_h1=256, expert_h2=64, dropout=0.05, interaction_dim=0,
                 use_cfi=False, use_pirp=True, use_hed=True, use_cag=True):
        super().__init__()
        self.use_cfi = use_cfi; self.use_pirp = use_pirp
        self.use_hed = use_hed; self.use_cag = use_cag
        self.input_norm = nn.LayerNorm(input_dim)
        self.gating = CAG(subsystem_dims, embed_dim, num_heads, num_experts, dropout)
        expert_input_dim = input_dim
        if use_hed:
            self.experts = build_diverse_experts(expert_input_dim, num_experts, expert_h1, expert_h2, dropout)
        else:
            self.experts = nn.ModuleList([ResidualExpert(expert_input_dim, expert_h1, expert_h2, dropout) for _ in range(num_experts)])
        if use_pirp:
            self.pirp = PhysicsResidualPath(input_dim)
        self.num_experts = num_experts

    def forward(self, x, subs):
        xn = self.input_norm(x)
        g = self.gating(subs)
        preds = torch.stack([e(xn) for e in self.experts], dim=-1)
        moe = (g * preds).sum(dim=-1)
        if self.use_pirp:
            return self.pirp(xn, moe)
        return moe


# ── Feature / subsystem definitions ─────────────────────────────
FEATURES = [
    'Cement_kg','Silica_Fume_kg','Flyash_kg','Quartz_Powder_kg','GGBFS_kg',
    'Filler_kg','Sand_kg','Fiber_Amount','Fiber_Length_mm','Fiber_Diameter_mm',
    'Water_kg','SP_kg','Curing_Temp_C',
    'Total_Binder','WB_Ratio','Fiber_AR','Fiber_RI','SF_Cement_Ratio','SCM_Ratio',
    'Sand_Binder_Ratio','SP_Binder_Ratio','Is_Heat_Cured',
    'Fiber_Type_Encoded','Curing_Regime_Encoded',
    'Paste_Volume','Water_Cement_Ratio','Fiber_Volume_Frac','Binder_Intensity',
    'Specimen_MinDim','Cement_Grade','Sand_MaxSize',
]

SUBS = {
    'binder': ['Cement_kg','Silica_Fume_kg','Flyash_kg','GGBFS_kg','Total_Binder','WB_Ratio',
               'SF_Cement_Ratio','SCM_Ratio','SP_Binder_Ratio','Water_Cement_Ratio',
               'Binder_Intensity','Cement_Grade'],
    'fiber':  ['Fiber_Amount','Fiber_Length_mm','Fiber_Diameter_mm','Fiber_AR','Fiber_RI',
               'Fiber_Volume_Frac','Fiber_Type_Encoded'],
    'aggregate': ['Sand_kg','Quartz_Powder_kg','Filler_kg','Sand_Binder_Ratio','Sand_MaxSize'],
    'environment': ['Water_kg','SP_kg','Curing_Temp_C','Is_Heat_Cured','Paste_Volume','Curing_Regime_Encoded'],
    'testing': ['Specimen_MinDim'],
}
SUB_IDX = {n: [FEATURES.index(f) for f in fs] for n, fs in SUBS.items()}
SUBSYSTEM_DIMS = [len(SUB_IDX[k]) for k in sorted(SUB_IDX)]

MODEL_CONFIG = {
    'num_experts': 5, 'embed_dim': 64, 'num_heads': 1,
    'expert_h1': 256, 'expert_h2': 64, 'dropout': 0.05,
    'use_cfi': False, 'use_pirp': True, 'use_hed': True, 'use_cag': True,
    'interaction_dim': 0,
}

CO2_FACTORS = {
    "Cement": 0.894, "Silica Fume": 0.014, "Fly Ash": 0.009,
    "GGBFS": 0.031, "Quartz Powder": 0.040, "Filler": 0.016,
    "Sand": 0.002, "Steel Fiber": 1.500, "Water": 0.0003,
    "Superplasticizer": 0.720
}
COST_FACTORS = {
    "Cement": 82, "Silica Fume": 800, "Fly Ash": 40,
    "GGBFS": 100, "Quartz Powder": 800, "Filler": 120,
    "Sand": 9, "Steel Fiber": 1000, "Water": 1,
    "Superplasticizer": 3400
}


# ── Load model + normalizer ────────────────────────────────────
@st.cache_resource
def load_ensemble():
    """Load 5 production CompMoE-CFI models and the fitted normalizer."""
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    device = torch.device('cpu')  # CPU for web

    # Load portable normalizer (sklearn objects only, no custom class dependency)
    with open(os.path.join(ckpt_dir, 'normalizer_portable.pkl'), 'rb') as f:
        norm_data = pickle.load(f)

    class Norm:
        def __init__(self, x_scaler, y_scaler):
            self.x = x_scaler; self.y = y_scaler
        def tx(self, X):
            return self.x.transform(X).astype(np.float32)
        def inverse_y(self, y):
            return self.y.inverse_transform(y.reshape(-1, 1)).ravel()

    normalizer = Norm(norm_data['x_scaler'], norm_data['y_scaler'])

    models = []
    for i in range(5):
        m = CompMoE_CFI(
            input_dim=31, subsystem_dims=SUBSYSTEM_DIMS,
            **MODEL_CONFIG
        ).to(device)
        state = torch.load(os.path.join(ckpt_dir, f'surrogate_model_{i}.pt'),
                            map_location=device)
        m.load_state_dict(state)
        m.eval()
        models.append(m)

    return models, normalizer, device


# ── Prediction function ────────────────────────────────────────
def predict_strength(models, normalizer, device, raw_vals):
    """Run the actual CompMoE-CFI 5-model ensemble forward pass."""
    # Build 31-feature vector
    cement = raw_vals["Cement"]
    sf = raw_vals["Silica Fume"]
    fa = raw_vals["Fly Ash"]
    ggbfs = raw_vals["GGBFS"]
    qp = raw_vals.get("Quartz Powder", 0)
    filler = raw_vals.get("Filler", 0)
    sand = raw_vals["Sand"]
    fiber = raw_vals["Steel Fiber"]
    fl = raw_vals["Fiber Length"]
    fd = raw_vals["Fiber Diameter"]
    water = raw_vals["Water"]
    sp = raw_vals["SP"]
    curing = raw_vals["Curing Temp"]

    tb = cement + sf + fa + ggbfs
    tb_safe = max(tb, 1e-6)
    wb = water / tb_safe
    far = fl / max(fd, 1e-6)
    fri = fiber * far / 100.0
    sfc = sf / max(cement, 1e-6)
    scm = (tb - cement) / tb_safe
    sbr = sand / tb_safe
    spb = sp / tb_safe
    hc = 1.0 if curing > 50 else 0.0
    pv = (cement/3150 + sf/2200 + water/1000 + sp/1100) * 1000
    wcr = water / max(cement, 1e-6)
    fvf = fiber / 7850
    bi = tb / max(tb + sand + water, 1e-6)

    vec = np.array([
        cement, sf, fa, qp, ggbfs, filler, sand,
        fiber, fl, fd, water, sp, curing,
        tb, wb, far, fri, sfc, scm, sbr, spb, hc,
        0,  # Fiber_Type_Encoded (straight steel)
        0,  # Curing_Regime_Encoded (standard)
        pv, wcr, fvf, bi,
        50.0,   # Specimen_MinDim (50mm cube reference)
        52.5,   # Cement_Grade
        0.6,    # Sand_MaxSize
    ], dtype=np.float32).reshape(1, -1)

    # Normalize using the fitted QuantileTransformer + StandardScaler
    X_n = normalizer.tx(vec)
    X_t = torch.FloatTensor(X_n).to(device)
    sub_s = sorted(SUB_IDX.items())
    sub_inputs = [X_t[:, idx] for _, idx in sub_s]

    # 5-model ensemble average (actual forward pass, no_grad)
    preds = []
    with torch.no_grad():
        for m in models:
            p = m(X_t, sub_inputs).cpu().numpy()
            preds.append(p)

    avg_pred_sc = np.mean(preds, axis=0)
    strength = float(normalizer.inverse_y(avg_pred_sc)[0])
    return strength


# ── Helper functions ───────────────────────────────────────────
def compute_derived(vals):
    tb = vals["Cement"] + vals["Silica Fume"] + vals["Fly Ash"] + vals["GGBFS"]
    wb = vals["Water"] / tb if tb > 0 else 0
    sfc = vals["Silica Fume"] / vals["Cement"] if vals["Cement"] > 0 else 0
    scm = (tb - vals["Cement"]) / tb if tb > 0 else 0
    fvf = vals["Steel Fiber"] / 7850
    far = vals.get("Fiber Length", 13) / max(vals.get("Fiber Diameter", 0.2), 1e-6)
    return {"W/B": wb, "SF/C": sfc, "SCM Ratio": scm,
            "Fiber Vol %": fvf * 100, "Aspect Ratio": far, "Total Binder": tb}


def compute_co2(vals):
    total = 0; breakdown = {}
    mapping = {"Cement":"Cement","Silica Fume":"Silica Fume","Fly Ash":"Fly Ash",
               "GGBFS":"GGBFS","Sand":"Sand","Steel Fiber":"Steel Fiber",
               "Water":"Water","SP":"Superplasticizer"}
    for key, fk in mapping.items():
        if key in vals:
            co2 = vals[key] * CO2_FACTORS[fk]
            breakdown[fk] = co2; total += co2
    return total, breakdown


def compute_cost(vals):
    total = 0
    mapping = {"Cement":"Cement","Silica Fume":"Silica Fume","Fly Ash":"Fly Ash",
               "GGBFS":"GGBFS","Sand":"Sand","Steel Fiber":"Steel Fiber",
               "Water":"Water","SP":"Superplasticizer"}
    for key, fk in mapping.items():
        if key in vals:
            total += vals[key] * COST_FACTORS[fk] / 1000
    return total


def check_constraints(derived):
    warnings = []
    if derived["W/B"] < 0.14: warnings.append(("W/B ratio below 0.14", "error"))
    if derived["W/B"] > 0.25: warnings.append(("W/B ratio above 0.25", "error"))
    if derived["SF/C"] > 0.40: warnings.append(("SF/Cement ratio exceeds 0.40", "error"))
    if derived["SCM Ratio"] > 0.50: warnings.append(("SCM ratio exceeds 0.50", "warning"))
    if derived["Fiber Vol %"] > 4.0: warnings.append(("Fiber volume fraction exceeds 4%", "error"))
    return warnings


PARETO_SOLUTIONS = pd.DataFrame([
    {"Strategy":"Max Strength","Cement":919,"Silica Fume":115,"Fly Ash":1,"GGBFS":7,
     "Sand":1296,"Steel Fiber":176,"Water":149,"SP":30,"Strength":237,"CO2":1114,"Cost":583},
    {"Strategy":"Min CO₂","Cement":500,"Silica Fume":0,"Fly Ash":2,"GGBFS":1,
     "Sand":996,"Steel Fiber":39,"Water":121,"SP":20,"Strength":148,"CO2":515,"Cost":123},
    {"Strategy":"Best Efficiency","Cement":500,"Silica Fume":146,"Fly Ash":195,"GGBFS":1,
     "Sand":1265,"Steel Fiber":39,"Water":120,"SP":25,"Strength":194,"CO2":536,"Cost":586},
    {"Strategy":"Balanced Knee","Cement":692,"Silica Fume":128,"Fly Ash":16,"GGBFS":0,
     "Sand":1280,"Steel Fiber":40,"Water":121,"SP":22,"Strength":203,"CO2":690,"Cost":246},
])


# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="CompMoE — UHPC Prediction & Optimization",
    layout="wide", initial_sidebar_state="expanded"
)

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.markdown("## CompMoE-CFI")
st.sidebar.markdown("*UHPC Strength Prediction &\nSustainable Mix Optimization*")
st.sidebar.divider()
st.sidebar.markdown(
    "**Model:** 5-model ensemble\n\n"
    "**R²** = 0.961 · **MAE** = 4.48 MPa\n\n"
    "**Architecture:** CAG + PIRP + HED\n\n"
    "**Dataset:** 1,216 mixes"
)
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate", ["Strength Predictor", "Pareto Solutions", "About"],
    label_visibility="collapsed"
)

# ── PREDICTOR PAGE ─────────────────────────────────────────────
if page == "Strength Predictor":
    models, normalizer, device = load_ensemble()
    st.title("UHPC Compressive Strength Predictor")
    st.caption("Adjust mix proportions to predict 28-day compressive strength, embodied CO₂, and material cost using the production CompMoE-CFI ensemble.")

    col_input, col_results = st.columns([1.2, 1])

    with col_input:
        st.markdown("#### Binder")
        bc1, bc2 = st.columns(2)
        cement = bc1.slider("Cement (kg/m³)", 400, 1200, 700)
        sf = bc2.slider("Silica Fume (kg/m³)", 0, 300, 120)
        bc3, bc4 = st.columns(2)
        fa = bc3.slider("Fly Ash (kg/m³)", 0, 400, 50)
        ggbfs = bc4.slider("GGBFS (kg/m³)", 0, 300, 30)

        st.markdown("#### Fiber")
        fc1, fc2, fc3 = st.columns(3)
        fiber = fc1.slider("Steel Fiber (kg/m³)", 20, 250, 80)
        fiber_l = fc2.slider("Fiber Length (mm)", 6, 30, 13)
        fiber_d = fc3.slider("Fiber Diameter (mm)", 0.10, 0.50, 0.20, step=0.01)

        st.markdown("#### Aggregate")
        ac1, ac2 = st.columns(2)
        sand = ac1.slider("Sand (kg/m³)", 600, 1600, 1100)
        qp = ac2.slider("Quartz Powder (kg/m³)", 0, 500, 0)

        st.markdown("#### Environment")
        ec1, ec2, ec3 = st.columns(3)
        water = ec1.slider("Water (kg/m³)", 100, 250, 160)
        sp = ec2.slider("Superplasticizer (kg/m³)", 10, 60, 30)
        curing = ec3.slider("Curing Temp (°C)", 20, 90, 23)

    vals = {"Cement": cement, "Silica Fume": sf, "Fly Ash": fa, "GGBFS": ggbfs,
            "Sand": sand, "Steel Fiber": fiber, "Water": water, "SP": sp,
            "Fiber Length": fiber_l, "Fiber Diameter": fiber_d,
            "Quartz Powder": qp, "Filler": 0, "Curing Temp": curing}
    derived = compute_derived(vals)
    co2_total, co2_breakdown = compute_co2(vals)
    cost_total = compute_cost(vals)
    warnings = check_constraints(derived)

    # Actual model prediction
    strength_pred = predict_strength(models, normalizer, device, vals)

    with col_results:
        st.markdown("#### Predicted Performance")
        m1, m2, m3 = st.columns(3)
        m1.metric("Strength", f"{strength_pred:.0f} MPa",
                   delta=f"{strength_pred - 160:.0f} vs mean" if abs(strength_pred - 160) > 1 else None)
        m2.metric("CO₂", f"{co2_total:.0f} kg/m³",
                   delta=f"{co2_total - 1150:.0f} vs mean", delta_color="inverse")
        m3.metric("Cost", f"${cost_total:.0f}/m³",
                   delta=f"${cost_total - 400:.0f} vs mean", delta_color="inverse")

        st.markdown("#### Key Ratios")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("W/B", f"{derived['W/B']:.3f}")
        r2.metric("SF/C", f"{derived['SF/C']:.3f}")
        r3.metric("SCM %", f"{derived['SCM Ratio']:.1%}")
        r4.metric("Vf %", f"{derived['Fiber Vol %']:.2f}")

        for msg, level in warnings:
            if level == "error": st.error(msg)
            else: st.warning(msg)

        st.markdown("#### CO₂ Breakdown")
        co2_df = pd.DataFrame([
            {"Material": k, "CO₂ (kg)": v, "%": v/co2_total*100 if co2_total > 0 else 0}
            for k, v in co2_breakdown.items() if v > 0.1
        ]).sort_values("CO₂ (kg)", ascending=True)

        fig_co2 = go.Figure(go.Bar(
            x=co2_df["CO₂ (kg)"], y=co2_df["Material"], orientation="h",
            marker_color=["#1565C0" if m == "Cement" else "#616161" if m == "Steel Fiber"
                           else "#00ACC1" if m == "Superplasticizer" else "#81C784"
                           for m in co2_df["Material"]],
            text=[f"{v:.0f} ({p:.0f}%)" for v, p in zip(co2_df["CO₂ (kg)"], co2_df["%"])],
            textposition="auto"
        ))
        fig_co2.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0),
                               xaxis_title="kg CO₂-eq/m³", font=dict(size=12))
        st.plotly_chart(fig_co2, use_container_width=True)

        st.success("Prediction from the full CompMoE-CFI 5-model ensemble (R² = 0.961, MAE = 4.48 MPa)")


# ── PARETO PAGE ────────────────────────────────────────────────
elif page == "Pareto Solutions":
    st.title("Pareto-Optimal UHPC Mix Designs")
    st.caption("90 non-dominated solutions from NSGA-III optimization across strength, CO₂, and cost.")

    st.markdown("### Recommended Strategies")
    st.dataframe(
        PARETO_SOLUTIONS.style.format({
            "Strength":"{:.0f}","CO2":"{:.0f}","Cost":"{:.0f}",
            "Cement":"{:.0f}","Silica Fume":"{:.0f}","Fly Ash":"{:.0f}",
            "GGBFS":"{:.0f}","Sand":"{:.0f}","Steel Fiber":"{:.0f}",
            "Water":"{:.0f}","SP":"{:.0f}"}),
        use_container_width=True, hide_index=True
    )

    st.markdown("### Strategy Comparison")
    categories = ["Strength","CO₂ Reduction","Cost Reduction","Cement Efficiency","Fiber Efficiency"]
    fig_radar = go.Figure()
    colors = {"Max Strength":"#C62828","Min CO₂":"#2E7D32",
              "Best Efficiency":"#7B1FA2","Balanced Knee":"#E65100"}
    for _, row in PARETO_SOLUTIONS.iterrows():
        vals = [row["Strength"]/237*100, (1-row["CO2"]/1214)*100,
                (1-row["Cost"]/727)*100, (1-row["Cement"]/919)*100,
                (1-row["Steel Fiber"]/176)*100]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=categories+[categories[0]],
            fill="toself", name=row["Strategy"],
            line=dict(color=colors.get(row["Strategy"],"#333"), width=2),
            fillcolor=colors.get(row["Strategy"],"#333"), opacity=0.15))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100])),
        height=450, font=dict(size=12))
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("### CO₂ Breakdown by Strategy")
    co2_data = []
    for _, row in PARETO_SOLUTIONS.iterrows():
        for mat, factor in [("Cement",0.894),("Steel Fiber",1.5),("SP",0.72)]:
            co2_data.append({"Strategy":row["Strategy"],"Material":mat,
                              "CO₂":row.get(mat, row.get("SP",0))*factor})
    co2_df = pd.DataFrame(co2_data)
    fig_bd = px.bar(co2_df, x="Strategy", y="CO₂", color="Material", barmode="stack",
                     color_discrete_map={"Cement":"#1565C0","Steel Fiber":"#616161","SP":"#00ACC1"})
    fig_bd.update_layout(height=350, yaxis_title="CO₂ (kg CO₂-eq/m³)", font=dict(size=12))
    st.plotly_chart(fig_bd, use_container_width=True)


# ── ABOUT PAGE ─────────────────────────────────────────────────
elif page == "About":
    st.title("About CompMoE-CFI")
    st.markdown("""
    ### Architecture

    CompMoE-CFI decomposes UHPC input features into **five physically meaningful subsystems**
    (Binder, Fiber, Aggregate, Environment, Testing) and processes them through three
    novel mechanisms:

    - **CAG** — Compositional Attention Gating: multi-head cross-attention over subsystem
      embeddings generates expert routing weights
    - **HED** — Hierarchical Expert Diversity: 5 heterogeneous experts (1 linear, 2 shallow,
      2 residual) matched to compositional complexity
    - **PIRP** — Physics-Informed Residual Path: learned linear skip connection (α ≈ 0.57)
      that decomposes prediction into linear and nonlinear components

    ### Performance (Production Ensemble)

    | Metric | Value |
    |--------|-------|
    | R² (production) | 0.961 |
    | MAE | 4.48 MPa |
    | RMSE | 6.72 MPa |
    | R² (25-eval CV) | 0.895 ± 0.019 |
    | Dataset | 1,216 mixes from ~130 studies |

    ### CO₂ and Cost References

    - **CO₂:** Wang et al. (2024) *Materials* 17(7):1670, Table 3
    - **Cost:** Wakjira et al. (2024) *Constr. Build. Mater.* 416:135114, Table 2

    ### Dataset Source

    Malik, U.J., Mohotti, D., Mo, H., & Lee, C.K. (2025). *A global dataset of UHPC mix
    designs with supplementary cementitious materials and nano additives*. Data in Brief, 58, 111235.

    ### Citation

    ```bibtex
    @article{compmoe2026,
      title={Compositional Mixture-of-Experts Neural Network for Compressive
             Strength Prediction and Sustainable Mix Optimization of
             Fiber-Reinforced Ultra-High Performance Concrete},
      year={2026}
    }
    ```
    """)
