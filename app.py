import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="AI Material Composition Designer",
    page_icon="🧪",
    layout="wide"
)

# =====================================================
# Load Models & Data (CACHED)
# =====================================================
@st.cache_resource
def load_models():
    xgb = joblib.load("global_xgb.pkl")
    knn = joblib.load("knn_property_space.pkl")
    return xgb, knn

@st.cache_data
def load_data():
    return pd.read_excel("FINAL ML PROPERTIES PREDICTION.xlsx")

xgb, knn = load_models()
df = load_data()

# Target property columns (must match training)
Y_COLS = [
    "Tensile (MPa)",
    "Modulus (GPa)",
    "Elongation (%)",
    "Impact (kJ/m²)"
]

# =====================================================
# Constants (same as training)
# =====================================================
MAX_TRAINED_MAH = 3.5
ERROR_WEIGHTS = np.array([1/70, 1/4, 1/150, 1/12])
LOCAL_PERTURB = [-5, -2.5, 0, 2.5, 5]

# =====================================================
# Inverse Design Function (Inference Only)
# =====================================================
def hybrid_inverse_design(
    target_properties,
    mah_fixed=5.0,
    pla_min=60,
    pla_max=90,
    top_k=5,
    error_tolerance=0.30
):
    # --- kNN search (property space)
    target_df = pd.DataFrame([target_properties], columns=Y_COLS)
    _, indices = knn.kneighbors(target_df)
    local_df = df.iloc[indices[0]].copy()

    # --- Enforce PLA constraint
    local_df = local_df[
        (local_df["PLA%"] >= pla_min) &
        (local_df["PLA%"] <= pla_max)
    ]

    if local_df.empty:
        return None, None, "No feasible compositions under PLA constraints"

    # --- Local composition exploration
    candidates = []
    for _, row in local_df.iterrows():
        pla = row["PLA%"]
        for delta in LOCAL_PERTURB:
            petg = row["PETG%"] + delta
            upvc = 100 - pla - petg
            if petg < 0 or upvc < 0:
                continue
            candidates.append([pla, petg, upvc, mah_fixed])

    candidates = np.unique(np.array(candidates), axis=0)
    if len(candidates) == 0:
        return None, None, "No valid candidates after optimization"

    # --- Feature engineering (MUST match training)
    cand_df = pd.DataFrame(
        candidates,
        columns=["PLA%", "PETG%", "UPVC%", "MA%"]
    )

    # Safety cap for MAH
    cand_df["MA%"] = np.minimum(cand_df["MA%"], MAX_TRAINED_MAH)

    cand_df["PLA_PETG"] = cand_df["PLA%"] * cand_df["PETG%"]
    cand_df["PLA_UPVC"] = cand_df["PLA%"] * cand_df["UPVC%"]
    cand_df["PETG_UPVC"] = cand_df["PETG%"] * cand_df["UPVC%"]
    cand_df["MA_present"] = (cand_df["MA%"] > 0).astype(int)

    model_features = [
        "PLA%", "PETG%", "UPVC%", "MA%",
        "PLA_PETG", "PLA_UPVC", "PETG_UPVC", "MA_present"
    ]

    preds = xgb.predict(cand_df[model_features])

    # --- Normalized error
    diffs = preds - np.array(target_properties)
    errors = np.sum(ERROR_WEIGHTS * (diffs ** 2), axis=1)

    if errors.min() > error_tolerance:
        return None, None, "Target not achievable within dataset support"

    top_idx = np.argsort(errors)[:top_k]
    return candidates[top_idx], preds[top_idx], errors[top_idx]

# =====================================================
# Sidebar — Inputs
# =====================================================

st.sidebar.title("🎯 Target Mechanical Properties")

tensile = st.sidebar.number_input(
    "Tensile Strength (MPa)",
    min_value=35.0,
    max_value=75.0,
    value=60.0,
    step=0.1
)

modulus = st.sidebar.number_input(
    "Elastic Modulus (GPa)",
    min_value=1.5,
    max_value=4.5,
    value=3.2,
    step=0.01
)

elongation = st.sidebar.number_input(
    "Elongation at Break (%)",
    min_value=2.0,
    max_value=150.0,
    value=20.0,
    step=0.1
)

impact = st.sidebar.number_input(
    "Impact Strength (kJ/m²)",
    min_value=1.0,
    max_value=12.0,
    value=4.0,
    step=0.1
)

st.sidebar.markdown("---")
st.sidebar.info(
    "🔒 **Constraints**\n"
    "- PLA: 60–90%\n"
    "- MAH: 5% (conceptual)\n"
    "- PETG & UPVC optimized automatically"
)

run = st.sidebar.button("🚀 Generate Optimal Blends")

# =====================================================
# Main UI
# =====================================================
st.title("🧪 AI-Driven Material Composition Designer")
st.caption(
    "Inverse-design system that predicts optimal PLA–PETG–UPVC compositions "
    "for desired mechanical properties before physical experimentation."
)

if run:
    target = [tensile, modulus, elongation, impact]
    comps, props, errs = hybrid_inverse_design(target)

    if comps is None:
        st.error(errs)
    else:
        st.success("✅ Optimal compositions found")

        rows = []
        for i in range(len(comps)):
            rows.append({
                "PLA (%)": comps[i][0],
                "PETG (%)": comps[i][1],
                "UPVC (%)": comps[i][2],
                "MAH (%)": comps[i][3],
                "Tensile (MPa)": props[i][0],
                "Modulus (GPa)": props[i][1],
                "Elongation (%)": props[i][2],
                "Impact (kJ/m²)": props[i][3],
                "Error Score": errs[i]
            })

        result_df = pd.DataFrame(rows)

        st.subheader("🔝 Top Recommended Compositions")
        st.dataframe(result_df.style.format("{:.2f}"), use_container_width=True)

        st.subheader("📊 Target vs Best Prediction")
        compare_df = pd.DataFrame({
            "Property": ["Tensile", "Modulus", "Elongation", "Impact"],
            "Target": target,
            "Predicted": props[0]
        })

        st.bar_chart(compare_df.set_index("Property"))

        st.info(
            "🧠 **How it works**\n"
            "- kNN retrieves experimentally similar materials\n"
            "- XGBoost refines predictions smoothly\n"
            "- Constraints ensure sustainability & manufacturability"
        )

# =====================================================
# Footer
# =====================================================
st.markdown("---")
st.caption("Production-ready ML inverse design system • Streamlit UI")
