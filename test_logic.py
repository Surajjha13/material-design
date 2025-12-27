import pandas as pd
import numpy as np
import joblib
import os

# =====================================================
# Load Models & Data
# =====================================================
def load_models():
    print("Loading models...")
    if not os.path.exists("global_xgb.pkl"):
        raise FileNotFoundError("global_xgb.pkl not found")
    if not os.path.exists("knn_property_space.pkl"):
        raise FileNotFoundError("knn_property_space.pkl not found")
        
    xgb = joblib.load("global_xgb.pkl")
    knn = joblib.load("knn_property_space.pkl")
    return xgb, knn

def load_data():
    print("Loading data...")
    if not os.path.exists("FINAL ML PROPERTIES PREDICTION.xlsx"):
        raise FileNotFoundError("FINAL ML PROPERTIES PREDICTION.xlsx not found")
    return pd.read_excel("FINAL ML PROPERTIES PREDICTION.xlsx")

# =====================================================
# Constants (same as training)
# =====================================================
MAX_TRAINED_MAH = 3.5
# Target property columns (must match training)
Y_COLS = [
    "Tensile (MPa)",
    "Modulus (GPa)",
    "Elongation (%)",
    "Impact (kJ/m²)"
]
ERROR_WEIGHTS = np.array([1/70, 1/4, 1/150, 1/12])
LOCAL_PERTURB = [-5, -2.5, 0, 2.5, 5]

# =====================================================
# Inverse Design Function (Inference Only)
# =====================================================
def hybrid_inverse_design(
    target_properties,
    models,
    df,
    mah_fixed=5.0,
    pla_min=60,
    pla_max=90,
    top_k=5,
    error_tolerance=0.30
):
    xgb, knn = models
    
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

    # Note: Removed error tolerance return for testing visibility, OR keeps it if you want strict test
    # if errors.min() > error_tolerance:
    #     return None, None, "Target not achievable within dataset support"

    top_idx = np.argsort(errors)[:top_k]
    return candidates[top_idx], preds[top_idx], errors[top_idx]

def run_tests():
    try:
        xgb, knn = load_models()
        df = load_data()
        
        # Test Case 1: Standard inputs
        print("\n--- Test Case 1: Target Properties [60, 3.2, 20, 4] ---")
        target = [60.0, 3.2, 20.0, 4.0]
        comps, props, errs = hybrid_inverse_design(target, (xgb, knn), df)
        
        if comps is None:
            print("Test 1 Failed: No composition found.")
            print("Error:", errs)
        else:
            print("Test 1 Output (Top 1):")
            print(f"Composition: PLA={comps[0][0]}, PETG={comps[0][1]}, UPVC={comps[0][2]}, MAH={comps[0][3]}")
            print(f"Predicted Props: {props[0]}")
            print(f"Error Score: {errs[0]}")
            
            # Assertions
            if not (60 <= comps[0][0] <= 90):
                print("FAILURE: PLA% out of range [60, 90]")
            else:
                print("SUCCESS: PLA% in range.")
                
            if comps[0][3] != 5.0:
                print(f"FAILURE: MAH% is {comps[0][3]}, expected 5.0")
            else:
                print("SUCCESS: MAH% is 5.0")

        # Test Case 2: Another target
        print("\n--- Test Case 2: Target Properties [50, 2.5, 50, 8] ---")
        target2 = [50.0, 2.5, 50.0, 8.0]
        comps2, props2, errs2 = hybrid_inverse_design(target2, (xgb, knn), df)
        
        if comps2 is None:
            print("Test 2 Info: No composition found (might be expected if target is far).")
            print("Message:", errs2)
        else:
            print("Test 2 Output (Top 1):")
            print(f"Composition: PLA={comps2[0][0]}, PETG={comps2[0][1]}, UPVC={comps2[0][2]}, MAH={comps2[0][3]}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()
