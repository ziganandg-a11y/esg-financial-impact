import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
BASE_FILE = "esg_financial_dataset_with_sector.xlsx"  # put this file next to app.py

st.set_page_config(page_title="ESG – EBITDA Impact Tool", layout="wide")

# -----------------------------
# LOAD BASE DATA (INTERNAL ONLY)
# -----------------------------
try:
    base_df = pd.read_excel(BASE_FILE)
except FileNotFoundError:
    st.error(
        f"Internal model data file '{BASE_FILE}' not found. "
        "Place it in the same folder as app.py."
    )
    st.stop()


def build_numeric_df(df):
    """Return numeric df with NA filled by median."""
    numeric = df.select_dtypes(include=["number"]).copy()
    for col in numeric.columns:
        median_val = numeric[col].median()
        numeric[col] = numeric[col].fillna(median_val)
    return numeric


def filter_by_sectors(df, selected_sectors):
    if "Sector" in df.columns and selected_sectors:
        return df[df["Sector"].isin(selected_sectors)]
    return df


# -----------------------------
# INIT SESSION STATE
# -----------------------------
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False

if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False

if "selected_sectors" not in st.session_state:
    # default => all sectors (or None if no sector column)
    if "Sector" in base_df.columns:
        st.session_state.selected_sectors = list(base_df["Sector"].dropna().unique())
    else:
        st.session_state.selected_sectors = None

if "target_col" not in st.session_state:
    st.session_state.target_col = None

if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None


st.title("ESG Financial Impact – EBITDA Margin Prediction")

# ============================================================
# STEP 1 – MODEL SETUP (SECTORS + TARGET + PREDICTORS)
# ============================================================
st.header("Model setup")

# sector selection for internal model
if "Sector" in base_df.columns:
    sectors = sorted(base_df["Sector"].dropna().unique())
    selected_sectors = st.multiselect(
        "Select sectors to base the model on (internal training data)",
        options=sectors,
        default=st.session_state.selected_sectors or sectors,
    )
else:
    selected_sectors = None
    st.info("No sector information found in internal data; model will use all observations.")

# build numeric data based on chosen sectors
filtered_base = filter_by_sectors(base_df, selected_sectors)
numeric_df = build_numeric_df(filtered_base)

if numeric_df.empty:
    st.error("No numeric data left after filtering. Adjust sector selection.")
    st.stop()

# auto-detect EBITDA target
target_candidates = [
    c for c in numeric_df.columns
    if "ebitda" in c.lower() and "margin" in c.lower()
]
if not target_candidates:
    target_candidates = [c for c in numeric_df.columns if "ebitda" in c.lower()]
default_target = target_candidates[0] if target_candidates else numeric_df.columns[0]

# UI: choose target
target_col = st.selectbox(
    "Choose target variable (typically an EBITDA margin measure)",
    options=numeric_df.columns.tolist(),
    index=numeric_df.columns.tolist().index(
        st.session_state.target_col or default_target
    ),
)

# predictors
predictor_candidates = [c for c in numeric_df.columns if c != target_col]
default_features = (
    st.session_state.feature_cols
    if st.session_state.feature_cols
    else predictor_candidates
)

feature_cols = st.multiselect(
    "Choose predictor variables for the internal model",
    options=predictor_candidates,
    default=default_features,
)

if not feature_cols:
    st.warning("Select at least one predictor variable to continue.")

# detect ESG-like variables (for later scenarios)
esg_cols = [
    c for c in numeric_df.columns
    if ("esg" in c.lower()) or ("sustain" in c.lower()) or ("csr" in c.lower())
]

# button to move on to input step
if st.button("Continue to data input"):
    if not feature_cols:
        st.warning("Please select at least one predictor before continuing.")
    else:
        st.session_state.model_ready = True
        st.session_state.selected_sectors = selected_sectors
        st.session_state.target_col = target_col
        st.session_state.feature_cols = feature_cols
        st.session_state.analysis_run = False  # reset if model changed

# if model not configured yet => stop here
if not st.session_state.model_ready:
    st.stop()

# ============================================================
# STEP 2 – COMPANY / USER INPUT (SCENARIO DATA)
# ============================================================
st.header("Provide your own data (scenario input)")

st.write("You can either fill in the fields manually or upload a CSV/Excel file using the same column names.")

# ---- manual input (single row) ----
col_esg, col_pe = st.columns(2)
with col_esg:
    esg_rating = st.slider(
        "ESG_Rating_Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.1,
        help="Drag to set the ESG rating score (0–100).",
    )
with col_pe:
    pe_ratio = st.number_input("P/E Ratio", value=15.0)

col_rev, col_ebitda = st.columns(2)
with col_rev:
    revenue = st.number_input("Revenue", value=1_000_000.0, step=10_000.0, format="%.2f")
with col_ebitda:
    ebitda = st.number_input("EBITDA", value=150_000.0, step=10_000.0, format="%.2f")

col_ni, col_eps = st.columns(2)
with col_ni:
    net_income = st.number_input("Net Income", value=100_000.0, step=5_000.0, format="%.2f")
with col_eps:
    eps = st.number_input("EPS", value=2.5, step=0.1, format="%.3f")

col_mc, col_debt = st.columns(2)
with col_mc:
    market_cap = st.number_input("Market Cap", value=5_000_000.0, step=50_000.0, format="%.2f")
with col_debt:
    debt_to_equity = st.number_input("Debt to Equity", value=0.5, step=0.05, format="%.3f")

col_cr, col_qr = st.columns(2)
with col_cr:
    current_ratio = st.number_input("Current Ratio", value=1.5, step=0.1, format="%.3f")
with col_qr:
    quick_ratio = st.number_input("Quick Ratio", value=1.2, step=0.1, format="%.3f")

col_ret, col_vol = st.columns(2)
with col_ret:
    return_1y = st.number_input("1Y Return", value=5.0, step=0.5, format="%.3f")
with col_vol:
    avg_vol_30d = st.number_input("Avg Volatility (30D)", value=10.0, step=0.5, format="%.3f")

ebitda_margin_manual = st.number_input(
    "EBITDA Margin",
    value=15.0,
    step=0.5,
    format="%.3f",
    help="Optional: if you already know your EBITDA margin.",
)

manual_df = pd.DataFrame(
    {
        "ESG_Rating_Score": [esg_rating],
        "P/E Ratio": [pe_ratio],
        "Revenue": [revenue],
        "EBITDA": [ebitda],
        "Net Income": [net_income],
        "EPS": [eps],
        "Market Cap": [market_cap],
        "Debt to Equity": [debt_to_equity],
        "Current Ratio": [current_ratio],
        "Quick Ratio": [quick_ratio],
        "1Y Return": [return_1y],
        "Avg Volatility (30D)": [avg_vol_30d],
        "EBITDA Margin": [ebitda_margin_manual],
    }
)

st.write("Preview of manually entered data:")
st.dataframe(manual_df)

scenario_df = manual_df.copy()

st.subheader("Or upload a scenario file (optional)")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel with the same column names",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is not None:
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            scenario_df = pd.read_csv(uploaded_file)
        else:
            scenario_df = pd.read_excel(uploaded_file)

        st.success("Scenario file loaded.")
        st.dataframe(scenario_df.head())
        st.caption(
            "Make sure the column names in your file match the model variables exactly "
            "(including spelling and spaces)."
        )
    except Exception as e:
        st.error(f"Error reading file: {e}")
        scenario_df = manual_df.copy()

st.markdown("---")
st.header("Run analysis")

if st.button("Run ESG–EBITDA analysis"):
    st.session_state.analysis_run = True

if not st.session_state.analysis_run:
    st.info("Click the button above to run the analysis with your input.")
    st.stop()

# ============================================================
# STEP 3 – TRAIN MODEL (INTERNAL) & SHOW RESULTS
# ============================================================
# Rebuild training data based on stored configuration
filtered_base = filter_by_sectors(base_df, st.session_state.selected_sectors)
numeric_df = build_numeric_df(filtered_base)

target_col = st.session_state.target_col
feature_cols = st.session_state.feature_cols

X_train = numeric_df[feature_cols]
y_train = numeric_df[target_col]

model = LinearRegression()
model.fit(X_train, y_train)
r2 = model.score(X_train, y_train)

st.header("Results & insights")

st.metric("Model R² (goodness of fit)", f"{r2:.3f}")

# re-detect ESG-like columns (based on training data)
esg_cols = [
    c for c in numeric_df.columns
    if ("esg" in c.lower()) or ("sustain" in c.lower()) or ("csr" in c.lower())
]

# ------------------------------------------------
# RESULTS TABS – USERS CHOOSE WHAT THEY WANT TO SEE
# ------------------------------------------------
tab_scenario, tab_predictions, tab_visuals = st.tabs(
    ["ESG scenario", "Predictions for your data", "Visual insights"]
)

# --------------- TAB 1: ESG SCENARIO ---------------
with tab_scenario:
    st.subheader("ESG what-if scenario (based on internal model)")

    if esg_cols:
        esg_options = [c for c in feature_cols if c in esg_cols] or feature_cols
    else:
        esg_options = feature_cols

    esg_col = st.selectbox(
        "Select an ESG-related (or other) driver to vary",
        options=esg_options,
    )

    base_X = pd.DataFrame(X_train.mean()).T
    base_pred = model.predict(base_X)[0]

    st.write(
        f"Baseline predicted **{target_col}** (using average values in internal data): "
        f"**{base_pred:.2f}**"
    )

    current_esg_value = float(base_X[esg_col].iloc[0])

    esg_min = float(X_train[esg_col].min())
    esg_max = float(X_train[esg_col].max())
    if esg_min == esg_max:
        esg_min -= 1.0
        esg_max += 1.0

    new_esg_value = st.slider(
        f"Set a new value for {esg_col}",
        min_value=esg_min,
        max_value=esg_max,
        value=current_esg_value,
        step=(esg_max - esg_min) / 100 if esg_max > esg_min else 0.1,
    )

    scenario_X_base = base_X.copy()
    scenario_X_base[esg_col] = new_esg_value
    scenario_pred_base = model.predict(scenario_X_base)[0]
    delta_base = scenario_pred_base - base_pred

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Baseline EBITDA margin", f"{base_pred:.2f}")
    with col2:
        st.metric("Scenario EBITDA margin", f"{scenario_pred_base:.2f}")
    with col3:
        st.metric("Change", f"{delta_base:+.2f}")

    st.caption(
        "This scenario is based on the internal model. It shows how the predicted EBITDA margin "
        "would change if only the selected driver changes, holding all other factors at their average values."
    )

    # curve vs ESG
    st.markdown("**Model-implied curve**")

    esg_grid = np.linspace(esg_min, esg_max, 50)
    grid_X = pd.concat([base_X] * len(esg_grid), ignore_index=True)
    grid_X[esg_col] = esg_grid
    grid_pred = model.predict(grid_X)

    fig_curve, ax_curve = plt.subplots()
    ax_curve.plot(esg_grid, grid_pred, label="Predicted EBITDA margin")
    ax_curve.axvline(new_esg_value, linestyle="--", label="Selected value")
    ax_curve.set_xlabel(esg_col)
    ax_curve.set_ylabel(target_col)
    ax_curve.legend()
    st.pyplot(fig_curve)

# --------------- TAB 2: PREDICTIONS FOR USER DATA ---------------
with tab_predictions:
    st.subheader("Predicted EBITDA margin for your input")

    missing_cols = [c for c in feature_cols if c not in scenario_df.columns]
    if missing_cols:
        st.warning(
            "Your data is missing the following columns required for prediction: "
            + ", ".join(missing_cols)
        )
    else:
        scenario_X = scenario_df[feature_cols]
        scenario_pred = model.predict(scenario_X)

        scenario_out = scenario_df.copy()
        scenario_out["Predicted " + target_col] = scenario_pred
        st.dataframe(scenario_out)

        st.caption(
            f"Each row in your input is scored using the internal model. "
            f"The column 'Predicted {target_col}' shows the estimated EBITDA margin."
        )

# --------------- TAB 3: VISUAL INSIGHTS ---------------
with tab_visuals:
    st.subheader("Visual insights from the internal model data")

    col_left, col_right = st.columns(2)

    # ============================
    # LEFT COLUMN
    # ============================
    with col_left:
        st.markdown("**ESG vs EBITDA margin (scatter)**")

        # pick ESG variable for scatter
        esg_plot_col = None
        if "ESG_Rating_Score" in numeric_df.columns:
            esg_plot_col = "ESG_Rating_Score"
        elif esg_cols:
            esg_plot_col = esg_cols[0]

        if esg_plot_col is not None and esg_plot_col in numeric_df.columns:
            fig_esg, ax_esg = plt.subplots(figsize=(5,4))
            ax_esg.scatter(numeric_df[esg_plot_col], numeric_df[target_col], alpha=0.6)
            ax_esg.set_xlabel(esg_plot_col)
            ax_esg.set_ylabel(target_col)
            st.pyplot(fig_esg)
        else:
            st.info("No ESG-related numeric variable available for this plot.")

        # -------------------------
        # Bar chart
        # -------------------------
        st.markdown("**Average EBITDA margin by ESG level (binned)**")

        if esg_plot_col is not None and esg_plot_col in numeric_df.columns:
            df_bins = numeric_df[[esg_plot_col, target_col]].dropna().copy()

            # divide ESG into 5 bins
            df_bins["ESG_bin"] = pd.qcut(df_bins[esg_plot_col], q=5, duplicates="drop")

            mean_margin = (
                df_bins.groupby("ESG_bin")[target_col]
                .mean()
                .sort_index()
            )

            fig_bar, ax_bar = plt.subplots(figsize=(5,4))
            ax_bar.bar(range(len(mean_margin)), mean_margin.values, width=0.7)
            ax_bar.set_xticks(range(len(mean_margin)))
            ax_bar.set_xticklabels([str(b) for b in mean_margin.index], rotation=45, ha="right")
            ax_bar.set_ylabel(target_col)
            ax_bar.set_xlabel(esg_plot_col + " (binned)")
            st.pyplot(fig_bar)

            st.caption(
                "Bars show the **average EBITDA margin** in different ESG ranges "
                "(ESG values split into quantile-based groups). "
                "This avoids the zig-zag effect caused by line charts."
            )
        else:
            st.info("Cannot create ESG bins because no ESG variable was detected.")

    # ============================
    # RIGHT COLUMN
    # ============================
    with col_right:
        st.markdown("**Actual vs predicted EBITDA margin**")

        y_pred_train = model.predict(X_train)

        fig_ap, ax_ap = plt.subplots(figsize=(5,4))
        ax_ap.scatter(y_train, y_pred_train, alpha=0.6)
        ax_ap.set_xlabel("Actual " + target_col)
        ax_ap.set_ylabel("Predicted " + target_col)

        # diagonal reference
        min_val = min(float(y_train.min()), float(y_pred_train.min()))
        max_val = max(float(y_train.max()), float(y_pred_train.max()))
        ax_ap.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        st.pyplot(fig_ap)

    # ============================
    # CORRELATION HEATMAP BELOW COLUMNS
    # ============================
    st.markdown("**Correlation heatmap of model variables**")
    corr = numeric_df[feature_cols + [target_col]].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(7,4))
    cax = ax_corr.matshow(corr)
    fig_corr.colorbar(cax)
    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_yticks(range(len(corr.columns)))
    ax_corr.set_xticklabels(corr.columns, rotation=90)
    ax_corr.set_yticklabels(corr.columns)
    st.pyplot(fig_corr)
