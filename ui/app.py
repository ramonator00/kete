import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import applications_to_frame, generate_demo_applications
from src.data.preprocessing import ID_COLUMN, TARGET_COLUMN
from src.domain import Action, FeedbackReason
from src.evolutionary.policy import PolicyParameters, recommend_action
from src.feedback.feedback_repository import FeedbackRepository
from src.feedback.feedback_service import FeedbackService
from src.ml.explain import explain_features, explain_prediction
from src.ml.model_io import load_metadata, load_model, model_exists
from src.ml.predict import predict, predict_from_features


st.set_page_config(page_title="KETE Credit Decision Support", layout="wide")

NAVY = "#1D3458"
CHAMPAGNE = "#C7AA6A"
WARM_WHITE = "#FAF8F3"
INK = "#111111"
MUTED = "#5F6368"
RISK_RED = "#B54747"
RISK_GREEN = "#2F7D59"

FEATURE_LABELS = {
    "LIMIT_BAL": "Credit limit",
    "SEX": "Sex",
    "EDUCATION": "Education",
    "MARRIAGE": "Marital status",
    "AGE": "Age",
    "PAY_0": "Latest repayment status",
    "PAY_2": "Repayment status 2 months ago",
    "PAY_3": "Repayment status 3 months ago",
    "PAY_4": "Repayment status 4 months ago",
    "PAY_5": "Repayment status 5 months ago",
    "PAY_6": "Repayment status 6 months ago",
    "BILL_AMT1": "Latest bill amount",
    "BILL_AMT2": "Bill amount 2 months ago",
    "BILL_AMT3": "Bill amount 3 months ago",
    "BILL_AMT4": "Bill amount 4 months ago",
    "BILL_AMT5": "Bill amount 5 months ago",
    "BILL_AMT6": "Bill amount 6 months ago",
    "PAY_AMT1": "Latest payment amount",
    "PAY_AMT2": "Payment amount 2 months ago",
    "PAY_AMT3": "Payment amount 3 months ago",
    "PAY_AMT4": "Payment amount 4 months ago",
    "PAY_AMT5": "Payment amount 5 months ago",
    "PAY_AMT6": "Payment amount 6 months ago",
    "debt_to_income_ratio": "Debt-to-income ratio",
    "late_payments_12m": "Late payments in last 12 months",
    "credit_amount_to_income": "Requested credit relative to income",
    "repayment_history_score": "Repayment history score",
    "employment_tenure": "Employment tenure",
    "collateral_coverage": "Collateral coverage",
}

SEX_LABELS = {1: "Male", 2: "Female"}
EDUCATION_LABELS = {
    0: "Unknown",
    1: "Graduate school",
    2: "University",
    3: "High school",
    4: "Other",
    5: "Unknown",
    6: "Unknown",
}
MARRIAGE_LABELS = {0: "Unknown", 1: "Married", 2: "Single", 3: "Other"}

FEEDBACK_REASON_LABELS = {
    FeedbackReason.ADDITIONAL_COLLATERAL: "Additional collateral provided",
    FeedbackReason.INCOME_STABILITY: "Income stability concern",
    FeedbackReason.UNDECLARED_DEBT: "Undeclared debt identified",
    FeedbackReason.DATA_QUALITY: "Data quality issue",
    FeedbackReason.POLICY_CONSTRAINT: "Policy constraint applies",
    FeedbackReason.MODEL_UNCERTAINTY: "Model uncertainty too high",
    FeedbackReason.ADDITIONAL_INFORMATION: "Additional information received",
    FeedbackReason.MODEL_ERROR: "Suspected model error",
}


def apply_theme():
    st.markdown(
        f"""
        <style>
            :root {{
                --kete-navy: {NAVY};
                --kete-champagne: {CHAMPAGNE};
                --kete-warm-white: {WARM_WHITE};
                --kete-ink: {INK};
                --kete-muted: {MUTED};
                --kete-border: #E8E0D0;
                --kete-soft-gold: #EFE6D2;
            }}

            .stApp {{
                background: #FFFFFF;
                color: var(--kete-ink);
            }}

            [data-testid="stHeader"] {{
                background: var(--kete-navy) !important;
                border-bottom: 1px solid #152844;
            }}

            [data-testid="stHeader"] button,
            [data-testid="stHeader"] button *,
            [data-testid="stHeader"] svg,
            [data-testid="stHeader"] span,
            [data-testid="stHeader"] p {{
                color: #FFFFFF !important;
                fill: #FFFFFF !important;
                stroke: #FFFFFF !important;
            }}

            .block-container {{
                padding-top: 2.1rem;
                padding-bottom: 3rem;
                max-width: 1440px;
            }}

            h1, h2, h3, h4, h5, h6 {{
                color: var(--kete-ink);
                letter-spacing: 0;
            }}

            h1 {{
                font-size: 2.05rem;
                font-weight: 720;
                margin-bottom: 0.15rem;
            }}

            h2, h3 {{
                font-weight: 680;
            }}

            p, label, span, div {{
                color: var(--kete-ink);
                letter-spacing: 0;
            }}

            [data-testid="stCaptionContainer"] {{
                color: var(--kete-muted);
            }}

            .kete-header {{
                background: #FFFFFF;
                border: 1px solid var(--kete-border);
                border-top: 4px solid var(--kete-champagne);
                border-radius: 8px;
                padding: 1.2rem 1.3rem 1.05rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 14px 34px rgba(17, 17, 17, 0.045);
                color: var(--kete-ink);
            }}

            .kete-header-top {{
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 1rem;
            }}

            .kete-header-copy {{
                min-width: 0;
            }}

            .kete-github-link {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 38px;
                height: 38px;
                border-radius: 999px;
                border: 1px solid var(--kete-border);
                background: #FFFFFF;
                box-shadow: 0 8px 18px rgba(17, 17, 17, 0.04);
                text-decoration: none;
                flex: 0 0 auto;
            }}

            .kete-github-link:hover {{
                border-color: var(--kete-champagne);
                background: #FBFAF7;
            }}

            .kete-github-link svg {{
                width: 19px;
                height: 19px;
                fill: var(--kete-ink);
            }}


            .kete-header h1 {{
                color: var(--kete-ink);
                margin: 0;
            }}

            .kete-header p {{
                color: var(--kete-muted);
                margin: 0.3rem 0 0;
                font-size: 0.95rem;
            }}

            .kete-accent {{
                display: inline-block;
                width: 76px;
                height: 4px;
                margin-top: 0.85rem;
                background: var(--kete-champagne);
                border-radius: 999px;
            }}

            [data-testid="stMetric"] {{
                background: #FFFFFF;
                border: 1px solid var(--kete-border);
                border-radius: 8px;
                padding: 0.85rem 0.95rem;
                box-shadow: 0 10px 24px rgba(17, 17, 17, 0.035);
            }}

            [data-testid="stMetricLabel"] {{
                color: var(--kete-muted);
                font-size: 0.82rem;
            }}

            [data-testid="stMetricValue"] {{
                color: var(--kete-ink);
                font-weight: 720;
            }}

            div[data-baseweb="select"] > div,
            textarea,
            input {{
                border-color: var(--kete-border);
                background-color: #FFFFFF;
            }}

            div[data-baseweb="select"],
            div[data-baseweb="select"] *,
            div[data-baseweb="popover"],
            div[data-baseweb="popover"] *,
            ul[role="listbox"],
            ul[role="listbox"] *,
            li[role="option"],
            li[role="option"] * {{
                background-color: #FFFFFF !important;
                color: var(--kete-ink) !important;
            }}

            div[data-baseweb="popover"] {{
                border: 1px solid var(--kete-border) !important;
                border-radius: 8px !important;
                box-shadow: 0 18px 42px rgba(17, 17, 17, 0.12) !important;
                overflow: hidden;
            }}

            li[role="option"]:hover,
            li[aria-selected="true"] {{
                background-color: #FBFAF7 !important;
                color: var(--kete-ink) !important;
            }}

            div[data-testid="stDataFrame"] {{
                border: 1px solid var(--kete-border);
                border-radius: 8px;
                overflow: hidden;
                background: #FFFFFF !important;
                box-shadow: 0 10px 24px rgba(17, 17, 17, 0.025);
            }}

            div[data-testid="stDataFrame"] * {{
                color: var(--kete-ink);
            }}

            div[data-testid="stDataFrame"] [class*="glideDataEditor"],
            div[data-testid="stDataFrame"] canvas,
            div[data-testid="stDataFrame"] [role="grid"],
            div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {{
                background: #FFFFFF !important;
            }}

            div[data-testid="stDataFrame"] [class*="row"],
            div[data-testid="stDataFrame"] [class*="cell"],
            div[data-testid="stDataFrame"] [class*="table"] {{
                background-color: #FFFFFF !important;
            }}

            div[data-testid="stForm"] {{
                background: #FFFFFF;
                border: 1px solid var(--kete-border);
                border-top: 3px solid var(--kete-champagne);
                border-radius: 8px;
                padding: 1.15rem 1.15rem 1rem;
                box-shadow: 0 16px 38px rgba(17, 17, 17, 0.045);
            }}

            div[data-testid="stForm"] label,
            div[data-testid="stForm"] [data-testid="stWidgetLabel"] p {{
                color: var(--kete-ink) !important;
                font-weight: 650;
            }}

            div[data-testid="stRadio"] {{
                margin-bottom: 0.55rem;
            }}

            div[data-testid="stRadio"] label {{
                background: #FFFFFF;
                border: 1px solid var(--kete-border);
                border-radius: 999px;
                padding: 0.42rem 0.75rem 0.42rem 0.48rem;
                margin-right: 0.35rem;
                min-height: 36px;
            }}

            div[data-testid="stRadio"] label:hover {{
                border-color: var(--kete-champagne);
                background: #FBFAF7;
            }}

            div[data-testid="stRadio"] label * {{
                color: var(--kete-ink) !important;
            }}

            div[data-testid="stRadio"] [role="radiogroup"] {{
                gap: 0.35rem;
            }}

            div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {{
                border-color: var(--kete-champagne) !important;
                background-color: #FFFFFF !important;
            }}

            div[data-testid="stRadio"] [data-baseweb="radio"] [aria-checked="true"] + div,
            div[data-testid="stRadio"] [data-baseweb="radio"] div[aria-checked="true"] {{
                border-color: var(--kete-champagne) !important;
                background-color: var(--kete-champagne) !important;
            }}

            div[data-testid="stRadio"] label:has(input[value="APPROVE"]) {{
                border-color: #BFD7C8;
                background: #F7FBF8;
            }}

            div[data-testid="stRadio"] label:has(input[value="REVIEW"]) {{
                border-color: #E7C993;
                background: #FFF9EE;
            }}

            div[data-testid="stRadio"] label:has(input[value="DECLINE"]) {{
                border-color: #E5B8B8;
                background: #FFF7F6;
            }}

            div[data-testid="stRadio"] label:has(input[value="APPROVE"]) * {{
                color: #245D3B !important;
            }}

            div[data-testid="stRadio"] label:has(input[value="REVIEW"]) * {{
                color: #8A5A12 !important;
            }}

            div[data-testid="stRadio"] label:has(input[value="DECLINE"]) * {{
                color: #8E2F2F !important;
            }}

            div[data-testid="stCheckbox"] label {{
                background: #FBFAF7;
                border: 1px solid var(--kete-border);
                border-radius: 8px;
                padding: 0.58rem 0.72rem;
                width: fit-content;
            }}

            div[data-testid="stCheckbox"] label * {{
                color: var(--kete-ink) !important;
            }}

            div[data-testid="stCheckbox"] [data-baseweb="checkbox"] {{
                border-color: var(--kete-champagne) !important;
            }}

            div[data-baseweb="tag"] {{
                background-color: #FBFAF7 !important;
                border: 1px solid var(--kete-border) !important;
                color: var(--kete-ink) !important;
            }}

            div[data-baseweb="tag"] * {{
                color: var(--kete-ink) !important;
            }}

            textarea {{
                background: #FFFFFF !important;
                color: var(--kete-ink) !important;
                border: 1px solid var(--kete-border) !important;
                border-radius: 8px !important;
                min-height: 96px !important;
            }}

            textarea::placeholder {{
                color: #8F8779 !important;
                opacity: 1 !important;
            }}

            div[data-baseweb="textarea"] {{
                background: #FFFFFF !important;
                border-color: var(--kete-border) !important;
            }}

            div[data-baseweb="textarea"] * {{
                background: #FFFFFF !important;
                color: var(--kete-ink) !important;
            }}

            button[kind="primary"],
            div.stButton > button,
            div[data-testid="stFormSubmitButton"] button {{
                background: var(--kete-navy);
                border: 1px solid var(--kete-navy);
                color: #FFFFFF;
                border-radius: 6px;
                padding: 0.55rem 1.05rem;
                font-weight: 650;
                box-shadow: 0 10px 22px rgba(29, 52, 88, 0.18);
            }}

            div[data-testid="stFormSubmitButton"] button * {{
                color: #FFFFFF !important;
            }}

            div.stButton > button:hover,
            div[data-testid="stFormSubmitButton"] button:hover {{
                background: #152844;
                border-color: #152844;
                color: #FFFFFF;
            }}

            button[role="tab"] {{
                color: var(--kete-muted);
                font-weight: 650;
            }}

            button[role="tab"][aria-selected="true"] {{
                color: var(--kete-ink);
                border-bottom: 2px solid var(--kete-champagne) !important;
            }}

            div[data-baseweb="tab-highlight"] {{
                background-color: var(--kete-champagne) !important;
                height: 2px;
            }}

            div[data-baseweb="tab-list"] button[aria-selected="true"] * {{
                color: var(--kete-ink) !important;
            }}

            div[data-baseweb="tab-list"] [style*="background-color"] {{
                background-color: var(--kete-champagne) !important;
            }}

            div[data-baseweb="tab-border"] {{
                background-color: #EFE8DA !important;
            }}

            .kete-feature-table {{
                background: #FFFFFF;
                border: 1px solid var(--kete-border);
                border-radius: 8px;
                box-shadow: 0 14px 34px rgba(17, 17, 17, 0.045);
                overflow: hidden;
            }}

            .kete-feature-table-header,
            .kete-feature-row {{
                display: grid;
                grid-template-columns: minmax(190px, 1.3fr) minmax(130px, 0.8fr) minmax(95px, 0.55fr);
                gap: 0;
            }}

            .kete-feature-table-header.kete-two-col,
            .kete-feature-row.kete-two-col {{
                grid-template-columns: minmax(190px, 1.2fr) minmax(130px, 0.8fr);
            }}

            .kete-feature-table-header {{
                background: #FBFAF7;
                border-bottom: 1px solid var(--kete-border);
            }}

            .kete-feature-scroll {{
                max-height: 430px;
                overflow-y: auto;
            }}

            .kete-feature-cell {{
                padding: 0.72rem 0.78rem;
                border-right: 1px solid #EFE8DA;
                color: var(--kete-ink);
                font-size: 0.89rem;
                line-height: 1.25;
            }}

            .kete-feature-cell:last-child {{
                border-right: 0;
            }}

            .kete-feature-heading {{
                color: #6F6759;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }}

            .kete-feature-row {{
                border-bottom: 1px solid #F0EADF;
                background: #FFFFFF;
            }}

            .kete-feature-row:nth-child(even) {{
                background: #FDFCFA;
            }}

            .kete-feature-row:last-child {{
                border-bottom: 0;
            }}

            .kete-feature-name {{
                font-weight: 650;
            }}

            .kete-feature-value {{
                font-variant-numeric: tabular-nums;
            }}

            .kete-feature-raw {{
                color: #81786A;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-size: 0.78rem;
            }}

            hr {{
                border-color: var(--kete-border);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_applications():
    return generate_demo_applications()


@st.cache_data
def load_uci_test_frame():
    path = ROOT / "data" / "processed" / "uci_test.csv"
    if not model_exists() or not path.exists():
        return None
    return pd.read_csv(path)


def recommend_from_probability(probability: float) -> Action:
    if probability >= 0.65:
        return Action.DECLINE
    if probability >= 0.45:
        return Action.REVIEW
    return Action.APPROVE


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def format_money(value) -> str:
    return f"NT$ {float(value):,.0f}"


def format_repayment_status(value) -> str:
    value = int(value)
    if value == -2:
        return "No consumption"
    if value == -1:
        return "Paid in full"
    if value == 0:
        return "Revolving credit / current"
    if value == 1:
        return "1 month payment delay"
    return f"{value} months payment delay"


def format_feature_value(feature: str, value) -> str:
    if feature == "SEX":
        return SEX_LABELS.get(int(value), f"Unknown ({int(value)})")
    if feature == "EDUCATION":
        return EDUCATION_LABELS.get(int(value), f"Unknown ({int(value)})")
    if feature == "MARRIAGE":
        return MARRIAGE_LABELS.get(int(value), f"Unknown ({int(value)})")
    if feature.startswith("PAY_"):
        return format_repayment_status(value)
    if feature.startswith("BILL_AMT") or feature.startswith("PAY_AMT") or feature == "LIMIT_BAL":
        return format_money(value)
    if feature == "AGE":
        return f"{int(value)} years"
    return str(value)


def explanation_frame(factors) -> pd.DataFrame:
    frame = pd.DataFrame([factor.__dict__ for factor in factors])
    frame["feature_label"] = frame["feature"].apply(feature_label)
    frame["risk_effect"] = frame["direction"].map(
        {
            "INCREASE": "Increases predicted default risk",
            "DECREASE": "Decreases predicted default risk",
        }
    )
    frame["signed_importance"] = frame.apply(
        lambda row: row["importance"] if row["direction"] == "INCREASE" else -row["importance"],
        axis=1,
    )
    return frame.sort_values("importance", ascending=True)


def render_feature_table(rows: list[dict], columns: Optional[list[tuple[str, str]]] = None, height: Optional[int] = None):
    columns = columns or [("Feature", "Feature"), ("Value", "Value"), ("Raw column", "Raw")]
    visible_columns = [(key, label) for key, label in columns if any(row.get(key, "") != "" for row in rows)]
    column_count = len(visible_columns)
    grid_template = " ".join(["minmax(120px, 1fr)"] * column_count)
    row_html = []
    for row in rows:
        cells = []
        for key, _label in visible_columns:
            css_class = "kete-feature-raw" if key.lower().startswith("raw") else ""
            cells.append(
                f'<div class="kete-feature-cell {css_class}">{escape(str(row.get(key, "")))}</div>'
            )
        row_html.append(
            f"""
            <div class="kete-feature-row">
                {''.join(cells)}
            </div>
            """
        )

    header_html = "".join(
        f'<div class="kete-feature-cell kete-feature-heading">{escape(label)}</div>'
        for _key, label in visible_columns
    )
    height = height or min(520, 58 + len(rows) * 45)
    components.html(
        f"""
        <style>
            :root {{
                --kete-ink: {INK};
                --kete-muted: {MUTED};
                --kete-border: #E8E0D0;
            }}
            body {{
                margin: 0;
                background: #FFFFFF;
                color: var(--kete-ink);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            .kete-feature-table {{
                background: #FFFFFF;
                border: 1px solid var(--kete-border);
                border-radius: 8px;
                box-shadow: 0 14px 34px rgba(17, 17, 17, 0.045);
                overflow: hidden;
            }}
            .kete-feature-table-header,
            .kete-feature-row {{
                display: grid;
                grid-template-columns: {grid_template};
                gap: 0;
            }}
            .kete-feature-table-header {{
                background: #FBFAF7;
                border-bottom: 1px solid var(--kete-border);
            }}
            .kete-feature-scroll {{
                max-height: 430px;
                overflow-y: auto;
            }}
            .kete-feature-cell {{
                padding: 0.72rem 0.78rem;
                border-right: 1px solid #EFE8DA;
                color: var(--kete-ink);
                font-size: 0.89rem;
                line-height: 1.25;
                box-sizing: border-box;
                overflow-wrap: anywhere;
            }}
            .kete-feature-cell:last-child {{
                border-right: 0;
            }}
            .kete-feature-heading {{
                color: #6F6759;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }}
            .kete-feature-row {{
                border-bottom: 1px solid #F0EADF;
                background: #FFFFFF;
            }}
            .kete-feature-row:nth-child(even) {{
                background: #FDFCFA;
            }}
            .kete-feature-row:last-child {{
                border-bottom: 0;
            }}
            .kete-feature-name {{
                font-weight: 650;
            }}
            .kete-feature-value {{
                font-variant-numeric: tabular-nums;
            }}
            .kete-feature-raw {{
                color: #81786A;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-size: 0.78rem;
            }}
        </style>
        <div class="kete-feature-table">
            <div class="kete-feature-table-header">
                {header_html}
            </div>
            <div class="kete-feature-scroll">
                {''.join(row_html)}
            </div>
        </div>
        """,
        height=height,
        scrolling=False,
    )


def render_explanation_chart(factors, title: str, caption: str):
    st.subheader(title)
    frame = explanation_frame(factors)
    fig = px.bar(
        frame,
        x="signed_importance",
        y="feature_label",
        color="risk_effect",
        orientation="h",
        labels={
            "signed_importance": "SHAP contribution",
            "feature_label": "Feature",
            "risk_effect": "Effect",
        },
        hover_data={"feature": True, "importance": ":.4f", "signed_importance": ":.4f"},
        color_discrete_map={
            "Increases predicted default risk": RISK_RED,
            "Decreases predicted default risk": RISK_GREEN,
        },
    )
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": INK, "size": 13},
        legend_title_text="",
        legend={
            "font": {"color": INK, "size": 12},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 10, "r": 20, "t": 10, "b": 10},
        xaxis={
            "gridcolor": "#F1EBDD",
            "zerolinecolor": CHAMPAGNE,
            "linecolor": "#E8E0D0",
            "tickfont": {"color": INK},
            "title": {"font": {"color": INK}},
        },
        yaxis={
            "title": None,
            "tickfont": {"color": INK, "size": 12},
        },
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
    )
    st.caption(caption)


def render_uci_application(row, metadata):
    application_id = f"UCI-{int(row[ID_COLUMN])}" if ID_COLUMN in row else "UCI"
    feature_columns = metadata["feature_columns"]
    features = row[feature_columns]
    prediction = predict_from_features(application_id, features)
    recommendation = recommend_from_probability(prediction.probability_of_default)
    factors = explain_features(features)

    left, middle, right = st.columns([1.2, 1, 1])
    with left:
        st.subheader(application_id)
        if TARGET_COLUMN in row:
            default_label = "Defaulted" if int(row[TARGET_COLUMN]) == 1 else "No default"
            st.caption(f"Historical test-set outcome: {default_label}")
        overview = [
            {
                "Feature": feature_label(column),
                "Value": format_feature_value(column, row[column]),
                "Raw column": column,
            }
            for column in feature_columns
        ]
        render_feature_table(overview)

    with middle:
        st.metric("Default probability", f"{prediction.probability_of_default:.1%}")
        st.metric("Risk category", prediction.risk_category)
        st.metric("Policy recommendation", recommendation.value)
        st.caption(f"Model {prediction.model_version} | CatBoost + SHAP")

    with right:
        render_explanation_chart(
            factors,
            "SHAP explanation",
            "Positive bars push the model toward higher default risk; negative bars push it toward lower risk. Values are on the model log-odds scale.",
        )

    render_feedback_form(application_id, recommendation)


def render_application(application):
    prediction = predict(application)
    recommendation = recommend_action(prediction, application, PolicyParameters())
    factors = explain_prediction(application)

    left, middle, right = st.columns([1.2, 1, 1])
    with left:
        st.subheader(application.application_id)
        overview = {
            "Age": application.age,
            "Gender": application.gender,
            "Credit amount": f"{application.credit_amount:,.0f}",
            "Annual income": f"{application.annual_income:,.0f}",
            "Existing debt": f"{application.existing_debt:,.0f}",
            "Debt / income": f"{application.debt_to_income_ratio:.2f}",
            "Late payments 12m": application.late_payments_12m,
            "Months employed": application.months_employed,
            "Collateral value": f"{application.collateral_value:,.0f}",
        }
        render_feature_table([{"Feature": key, "Value": str(value)} for key, value in overview.items()])

    with middle:
        st.metric("Default probability", f"{prediction.probability_of_default:.1%}")
        st.metric("Risk category", prediction.risk_category)
        st.metric("Policy recommendation", recommendation.value)
        st.caption(f"Model {prediction.model_version} | Policy generation_demo_00")

    with right:
        render_explanation_chart(
            factors,
            "Explanation",
            "Positive bars indicate risk-increasing factors; negative bars indicate risk-decreasing factors.",
        )

    render_feedback_form(application.application_id, recommendation)


def render_feedback_form(application_id: str, recommendation: Action):
    service = FeedbackService()
    st.divider()
    st.subheader("Advisor decision")
    with st.form("feedback_form", clear_on_submit=True):
        advisor_decision = st.radio(
            "Decision",
            [action.value for action in Action],
            index=[action.value for action in Action].index(recommendation.value),
            horizontal=True,
            label_visibility="collapsed",
        )
        reason_values = st.multiselect(
            "Structured override reasons",
            list(FeedbackReason),
            format_func=lambda reason: FEEDBACK_REASON_LABELS[reason],
            placeholder="Choose override reasons",
        )
        verified = st.checkbox("Supporting information is verified")
        comment = st.text_area("Advisor note", placeholder="Optional free-text note")
        submitted = st.form_submit_button("Save feedback")

    if submitted:
        try:
            record = service.submit(
                application_id=application_id,
                recommendation=recommendation,
                advisor_decision=Action(advisor_decision),
                reason_codes=reason_values,
                verified=verified,
                comment=comment,
            )
            st.success(f"Feedback saved. Immediate reward: {record.immediate_reward}")
        except ValueError as exc:
            st.error(str(exc))


def render_feedback_log():
    repository = FeedbackRepository()
    rows = repository.list_recent(25)
    st.subheader("Feedback loop")
    if not rows:
        st.info("No feedback recorded yet.")
        return
    frame = pd.DataFrame(rows)
    frame["reason_codes"] = frame["reason_codes"].apply(
        lambda items: ", ".join(FEEDBACK_REASON_LABELS.get(item, item.value) for item in items)
    )
    st.dataframe(frame, hide_index=True, width="stretch")


def compute_live_data_science_metrics(uci_test: Optional[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if uci_test is None or not model_exists():
        applications = generate_demo_applications()
        predictions = [predict(application) for application in applications]
        decisions = [recommend_action(prediction, application, PolicyParameters()).value for prediction, application in zip(predictions, applications)]
        frame = pd.DataFrame(
            {
                "decision": decisions,
                "probability": [prediction.probability_of_default for prediction in predictions],
                "gender": [application.gender for application in applications],
            }
        )
        metrics = {
            "Data source": "Synthetic demo applications",
            "Applications": len(frame),
            "Mean default probability": frame["probability"].mean(),
            "Approve rate": (frame["decision"] == Action.APPROVE.value).mean(),
            "Review rate": (frame["decision"] == Action.REVIEW.value).mean(),
            "Decline rate": (frame["decision"] == Action.DECLINE.value).mean(),
            "Feedback records": len(FeedbackRepository().list_recent(10_000)),
        }
        mix = frame["decision"].value_counts(normalize=True).rename_axis("Decision").reset_index(name="Share")
        return pd.DataFrame([metrics]), mix

    metadata = load_metadata()
    feature_columns = metadata["feature_columns"]
    model = load_model()
    probabilities = model.predict_proba(uci_test[feature_columns])[:, 1]
    decisions = [recommend_from_probability(float(probability)).value for probability in probabilities]

    live_frame = uci_test.copy()
    live_frame["probability"] = probabilities
    live_frame["decision"] = decisions
    y_true = live_frame[TARGET_COLUMN]

    feedback_count = len(FeedbackRepository().list_recent(10_000))
    approval_by_sex = live_frame.groupby("SEX")["decision"].apply(lambda values: (values == Action.APPROVE.value).mean())
    fairness_diff = float(approval_by_sex.max() - approval_by_sex.min()) if not approval_by_sex.empty else 0.0

    metrics = {
        "Data source": "Current CatBoost model on UCI test split",
        "Applications": len(live_frame),
        "Model version": metadata.get("model_version", "unknown"),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "Brier score": brier_score_loss(y_true, probabilities),
        "Mean default probability": live_frame["probability"].mean(),
        "Approve rate": (live_frame["decision"] == Action.APPROVE.value).mean(),
        "Review rate": (live_frame["decision"] == Action.REVIEW.value).mean(),
        "Decline rate": (live_frame["decision"] == Action.DECLINE.value).mean(),
        "Fairness diff. by sex": fairness_diff,
        "Feedback records": feedback_count,
        "Last refresh": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    mix = live_frame["decision"].value_counts(normalize=True).rename_axis("Decision").reset_index(name="Share")
    return pd.DataFrame([metrics]), mix


def format_metric_value(value) -> str:
    if isinstance(value, float):
        if 0 <= value <= 1:
            return f"{value:.1%}"
        return f"{value:.3f}"
    return str(value)


def render_experiments(uci_test: Optional[pd.DataFrame]):
    st.subheader("Live data science dashboard")
    metrics, decision_mix = compute_live_data_science_metrics(uci_test)
    metric_rows = [
        {
            "Metric": column,
            "Value": format_metric_value(metrics.iloc[0][column]),
        }
        for column in metrics.columns
    ]
    render_feature_table(
        metric_rows,
        columns=[
            ("Metric", "Metric"),
            ("Value", "Value"),
        ],
        height=430,
    )

    fig = px.bar(
        decision_mix,
        x="Decision",
        y="Share",
        color="Decision",
        color_discrete_map={
            Action.APPROVE.value: "#2F7D59",
            Action.REVIEW.value: "#C48A2C",
            Action.DECLINE.value: "#B54747",
        },
        labels={"Share": "Share of current applications"},
    )
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"color": INK, "size": 13},
        legend_title_text="",
        legend={"font": {"color": INK, "size": 12}},
        margin={"l": 10, "r": 20, "t": 20, "b": 10},
        xaxis={
            "title": None,
            "gridcolor": "#F1EBDD",
            "linecolor": "#E8E0D0",
            "tickfont": {"color": INK},
        },
        yaxis={
            "gridcolor": "#F1EBDD",
            "linecolor": "#E8E0D0",
            "tickfont": {"color": INK},
            "title": {"font": {"color": INK}},
            "tickformat": ".0%",
        },
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Live means recalculated from the currently loaded model, local UCI test split and current feedback database on each app refresh. Historical labels are used only for evaluation, not for making a decision."
    )


def main():
    apply_theme()
    st.markdown(
        """
        <div class="kete-header">
            <div class="kete-header-top">
                <div class="kete-header-copy">
                    <h1>KETE Credit Decision Support</h1>
                    <p>Explainable, human-supervised credit risk assistance. Recommendations are simulated and not real credit decisions.</p>
                </div>
                <a class="kete-github-link" href="https://github.com/ramonator00/kete" target="_blank" rel="noopener noreferrer" aria-label="Open GitHub repository">
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M12 0.5C5.65 0.5 0.5 5.65 0.5 12c0 5.09 3.29 9.39 7.86 10.91 0.58 0.1 0.79-0.25 0.79-0.56 0-0.28-0.01-1.02-0.02-2-3.2 0.7-3.88-1.54-3.88-1.54-0.52-1.33-1.28-1.69-1.28-1.69-1.05-0.72 0.08-0.7 0.08-0.7 1.16 0.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36 0.96 0.1-0.75 0.4-1.26 0.73-1.55-2.55-0.29-5.23-1.28-5.23-5.68 0-1.26 0.45-2.28 1.18-3.09-0.12-0.29-0.51-1.46 0.11-3.04 0 0 0.97-0.31 3.16 1.18 0.92-0.26 1.9-0.38 2.88-0.39 0.98 0 1.96 0.13 2.88 0.39 2.2-1.49 3.16-1.18 3.16-1.18 0.62 1.58 0.23 2.75 0.11 3.04 0.74 0.81 1.18 1.83 1.18 3.09 0 4.42-2.69 5.39-5.25 5.67 0.41 0.36 0.78 1.07 0.78 2.15 0 1.55-0.01 2.8-0.01 3.18 0 0.31 0.21 0.67 0.79 0.56A11.51 11.51 0 0 0 23.5 12C23.5 5.65 18.35 0.5 12 0.5Z"/>
                    </svg>
                </a>
            </div>
            <span class="kete-accent"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uci_test = load_uci_test_frame()

    advisor_tab, experiments_tab = st.tabs(["Advisor workflow", "Data science experiments"])
    with advisor_tab:
        if uci_test is not None:
            metadata = load_metadata()
            ids = uci_test[ID_COLUMN].astype(int).astype(str).tolist()
            selected_id = st.selectbox("Application", ids)
            row = uci_test.loc[uci_test[ID_COLUMN].astype(int).astype(str) == selected_id].iloc[0]
            render_uci_application(row, metadata)
        else:
            applications = load_applications()
            frame = applications_to_frame(applications)
            selected_id = st.selectbox("Application", frame["application_id"].tolist())
            application = next(item for item in applications if item.application_id == selected_id)
            render_application(application)
        render_feedback_log()

    with experiments_tab:
        render_experiments(uci_test)


if __name__ == "__main__":
    main()
