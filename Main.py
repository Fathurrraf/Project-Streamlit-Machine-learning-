import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ==========================================================
# 1. PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Bank Marketing Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. DESIGN SYSTEM / CSS
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #172B2A;
    --navy-soft: #203A37;
    --green: #237A62;
    --green-light: #E8F3EE;
    --orange: #D98A45;
    --red: #C65B55;
    --bg: #F3F6F5;
    --white: #FFFFFF;
    --text: #1D302C;
    --muted: #667873;
    --border: #DFE7E3;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1440px;
    padding-top: 2.1rem;
    padding-bottom: 3rem;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: var(--navy);
    border-right: 1px solid #29413D;
}

[data-testid="stSidebar"] > div {
    background: var(--navy);
}

[data-testid="stSidebar"] * {
    color: #EAF1ED;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #B8C9C1;
}

[data-testid="stSidebar"] hr {
    border-color: #38504A;
}

[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 5px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent;
    border-radius: 8px;
    padding: 9px 12px;
    border: 1px solid transparent;
    transition: 0.2s ease;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #29433D;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: #31564B;
    border: 1px solid #527568;
}

[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #EDF4F0 !important;
    font-weight: 600;
    font-size: 13px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none;
}

/* Typography */

h1, h2, h3, h4 {
    color: var(--text) !important;
    letter-spacing: -0.4px;
}

h1 {
    font-size: 30px !important;
    font-weight: 800 !important;
}

h2 {
    font-size: 22px !important;
    font-weight: 700 !important;
}

h3 {
    font-size: 17px !important;
    font-weight: 700 !important;
}

p, li {
    color: #425650;
    line-height: 1.65;
}

small {
    color: var(--muted);
}

/* Page header */

.page-eyebrow {
    color: var(--green);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.page-heading {
    color: var(--navy);
    font-size: 31px;
    line-height: 1.2;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 7px;
}

.page-description {
    color: #65766F;
    font-size: 14px;
    margin-bottom: 25px;
}

/* Metric cards */

.metric-card {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 19px;
    min-height: 112px;
    box-shadow: 0 2px 8px rgba(23, 43, 42, 0.025);
}

.metric-label {
    color: #64766F;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 10px;
}

.metric-value {
    color: #172B2A;
    font-size: 27px;
    font-weight: 800;
    line-height: 1.2;
}

.metric-note {
    color: #82918B;
    font-size: 11px;
    margin-top: 7px;
}

/* Content cards */

.content-card {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}

/* Prediction result */

.result-positive {
    background: #EAF5EF;
    border: 1px solid #B9D9C8;
    border-left: 5px solid #237A62;
    border-radius: 10px;
    padding: 20px;
}

.result-negative {
    background: #FFF1ED;
    border: 1px solid #F0C9BD;
    border-left: 5px solid #C65B55;
    border-radius: 10px;
    padding: 20px;
}

.result-title {
    font-size: 19px;
    font-weight: 800;
    margin-bottom: 6px;
}

.result-description {
    font-size: 13px;
    color: #53665F;
}

/* Section heading */

.section-heading {
    color: #203A34;
    font-size: 18px;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 4px;
}

.section-description {
    color: #788982;
    font-size: 12px;
    margin-bottom: 14px;
}

/* Dataframe */

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 9px;
    overflow: hidden;
}

/* Inputs */

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stDateInput input {
    border-radius: 8px;
}

.stButton button,
.stDownloadButton button,
[data-testid="stFormSubmitButton"] button {
    border-radius: 8px;
    font-weight: 700;
    min-height: 42px;
}

.stButton button[kind="primary"],
[data-testid="stFormSubmitButton"] button {
    background: var(--green);
    border-color: var(--green);
}

/* Tabs */

button[data-baseweb="tab"] {
    font-weight: 700;
    color: #61736B;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--green);
}

/* Alerts */

[data-testid="stAlert"] {
    border-radius: 9px;
}

/* Divider */

hr {
    border-color: var(--border);
    margin-top: 25px;
    margin-bottom: 25px;
}

/* Footer */

.footer {
    text-align: center;
    color: #899891;
    font-size: 11px;
    padding-top: 15px;
}

/* Explanation cards */

.calculation-card {
    background: #FFFFFF;
    border: 1px solid #DFE7E3;
    border-radius: 10px;
    padding: 17px;
    height: 100%;
}

.calculation-number {
    color: #237A62;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
}

.calculation-title {
    color: #203A34;
    font-size: 14px;
    font-weight: 800;
    margin-top: 7px;
    margin-bottom: 5px;
}

.calculation-description {
    color: #71817A;
    font-size: 12px;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. REUSABLE UI FUNCTIONS
# ==========================================================

def page_header(eyebrow, title, description):
    st.markdown(f"""
        <div class="page-eyebrow">{eyebrow}</div>
        <div class="page-heading">{title}</div>
        <div class="page-description">{description}</div>
    """, unsafe_allow_html=True)


def section_header(title, description=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )

    if description:
        st.markdown(
            f'<div class="section-description">{description}</div>',
            unsafe_allow_html=True
        )


def metric_card(label, value, note=""):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
    """, unsafe_allow_html=True)


def format_percent(value):
    return f"{value * 100:.2f}%"


def style_chart(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="DM Sans",
            color="#425650",
            size=12
        ),
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=False,
            linecolor="#DFE7E3"
        ),
        yaxis=dict(
            gridcolor="#EDF1EF",
            linecolor="#DFE7E3"
        )
    )
    return fig


def get_positive_probability(model, data):
    """
    Mengambil probabilitas kelas positif (1 / Yes)
    berdasarkan posisi kelas yang benar pada model.
    """
    probabilities = model.predict_proba(data)[0]
    classes = list(model.classes_)

    positive_index = classes.index(1)
    negative_index = classes.index(0)

    probability_yes = probabilities[positive_index]
    probability_no = probabilities[negative_index]

    return probability_yes, probability_no


def get_prediction_details(model, data):
    """
    Mengambil prediksi dan probabilitas dari model.
    """
    prediction = int(model.predict(data)[0])
    probability_yes, probability_no = get_positive_probability(
        model,
        data
    )

    return prediction, probability_yes, probability_no


# ==========================================================
# 4. LOAD DATASET
# ==========================================================

@st.cache_data
def load_data():

    filenames = ["bank.csv", "bank(2).csv"]

    for filename in filenames:
        try:
            data = pd.read_csv(filename, sep=";")

            if data.shape[1] == 1:
                data = pd.read_csv(filename)

            if "y" in data.columns:
                return data, filename

        except FileNotFoundError:
            continue

    return None, None


df_original, file_name = load_data()

if df_original is None:
    st.error(
        "Dataset tidak ditemukan. Letakkan bank.csv atau "
        "bank(2).csv dalam folder yang sama dengan Main.py."
    )
    st.stop()


# ==========================================================
# 5. DATA CLEANING
# ==========================================================

df = df_original.copy()
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].str.strip()

df["y"] = df["y"].astype(str).str.lower()

df = df[df["y"].isin(["yes", "no"])].copy()

jumlah_sebelum_duplikat = len(df)
df = df.drop_duplicates().reset_index(drop=True)
jumlah_duplikat = jumlah_sebelum_duplikat - len(df)

TARGET = "y"

# duration tidak digunakan karena nilainya baru diketahui
# setelah panggilan berlangsung.
FEATURES = [
    col for col in df.columns
    if col not in [TARGET, "duration"]
]

X = df[FEATURES].copy()
y = df[TARGET].map({"no": 0, "yes": 1})

numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# ==========================================================
# 6. TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================================
# 7. PREPROCESSING PIPELINE
# ==========================================================

def create_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])


# ==========================================================
# 8. MODEL TRAINING
# ==========================================================

@st.cache_resource
def train_models():

    model_definitions = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "SVM": SVC(
            probability=True,
            class_weight="balanced",
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
    }

    trained_models = {}
    result_rows = []

    for name, algorithm in model_definitions.items():

        pipeline = Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", algorithm)
        ])

        # Membatasi data training SVM agar tidak terlalu berat.
        if name == "SVM" and len(X_train) > 8000:

            sample_index = X_train.sample(
                n=8000,
                random_state=42
            ).index

            pipeline.fit(
                X_train.loc[sample_index],
                y_train.loc[sample_index]
            )

        else:
            pipeline.fit(X_train, y_train)

        prediction = pipeline.predict(X_test)

        probability = pipeline.predict_proba(X_test)[:, 1]

        result_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(
                y_test, prediction, zero_division=0
            ),
            "Recall": recall_score(
                y_test, prediction, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, prediction, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, probability)
        })

        trained_models[name] = pipeline

    results = pd.DataFrame(result_rows)

    results = results.sort_values(
        "F1 Score",
        ascending=False
    ).reset_index(drop=True)

    best_name = results.iloc[0]["Model"]

    return trained_models, results, best_name


# ==========================================================
# 9. SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("""
        <div style="padding: 12px 4px 18px 4px;">
            <div style="
                font-size: 12px;
                letter-spacing: 2px;
                font-weight: 800;
                color: #8DB8A5;">
                BANKING ANALYTICS
            </div>
            <div style="
                font-size: 23px;
                font-weight: 800;
                color: #FFFFFF;
                margin-top: 7px;">
                Bank Marketing
            </div>
            <div style="
                font-size: 11px;
                color: #B5C8BF;
                margin-top: 5px;">
                Machine Learning Dashboard
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<div style="font-size:10px;letter-spacing:1.5px;'
        'font-weight:800;color:#88A99A;margin-bottom:8px;">'
        'NAVIGATION</div>',
        unsafe_allow_html=True
    )

    menu = st.radio(
        "Navigasi",
        [
            "Beranda",
            "Tentang Dataset",
            "Eksplorasi Data",
            "Preprocessing",
            "Reduksi Dimensi (PCA)",
            "Modelling & Evaluasi",
            "Prediksi Satu Nasabah",
            "Prediksi Massal (CSV)",
            "Kesimpulan & Saran"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(f"""
        <div style="
            background:#203A37;
            border:1px solid #38534A;
            padding:13px;
            border-radius:9px;">
            <div style="font-size:10px;color:#A9C5B7;">
                DATASET AKTIF
            </div>
            <div style="
                font-size:13px;
                color:#FFFFFF;
                font-weight:700;
                margin-top:5px;">
                Bank Marketing
            </div>
            <div style="
                font-size:11px;
                color:#A9C5B7;
                margin-top:4px;">
                {file_name}
            </div>
        </div>
    """, unsafe_allow_html=True)


# ==========================================================
# 10. BERANDA
# ==========================================================

if menu == "Beranda":

    page_header(
        "OVERVIEW",
        "Dashboard Analisis Bank Marketing",
        "Pantau karakteristik nasabah, eksplorasi data, dan hasil "
        "prediksi potensi berlangganan deposito."
    )

    total_nasabah = len(df)
    total_yes = int((df["y"] == "yes").sum())
    total_no = int((df["y"] == "no").sum())
    conversion = total_yes / total_nasabah * 100

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "TOTAL NASABAH",
            f"{total_nasabah:,}",
            "Data setelah cleaning"
        )

    with c2:
        metric_card(
            "BERLANGGANAN",
            f"{total_yes:,}",
            "Target: Yes"
        )

    with c3:
        metric_card(
            "TIDAK BERLANGGANAN",
            f"{total_no:,}",
            "Target: No"
        )

    with c4:
        metric_card(
            "CONVERSION RATE",
            f"{conversion:.2f}%",
            "Proporsi target Yes"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])

    with left:

        section_header(
            "Distribusi Status Nasabah",
            "Perbandingan nasabah yang berlangganan dan tidak berlangganan."
        )

        target_data = df["y"].value_counts().reset_index()
        target_data.columns = ["Status", "Jumlah"]

        fig = px.pie(
            target_data,
            names="Status",
            values="Jumlah",
            hole=0.62,
            color="Status",
            color_discrete_map={
                "yes": "#237A62",
                "no": "#D98A45"
            }
        )

        fig.update_traces(
            textposition="outside",
            textinfo="label+percent"
        )

        st.plotly_chart(
            style_chart(fig, 370),
            use_container_width=True
        )

    with right:

        section_header(
            "Informasi Dataset",
            "Ringkasan sumber dan struktur data."
        )

        st.markdown(f"""
        <div class="content-card">
            <div style="font-size:12px;color:#71817A;">Nama Dataset</div>
            <div style="font-size:18px;font-weight:800;color:#172B2A;">
                Bank Marketing
            </div>
            <hr>
            <div style="display:flex;justify-content:space-between;">
                <span>Jumlah baris</span>
                <b>{len(df):,}</b>
            </div>
            <br>
            <div style="display:flex;justify-content:space-between;">
                <span>Jumlah kolom</span>
                <b>{len(df.columns)}</b>
            </div>
            <br>
            <div style="display:flex;justify-content:space-between;">
                <span>Fitur numerik</span>
                <b>{len(numeric_cols)}</b>
            </div>
            <br>
            <div style="display:flex;justify-content:space-between;">
                <span>Fitur kategorikal</span>
                <b>{len(categorical_cols)}</b>
            </div>
            <br>
            <div style="display:flex;justify-content:space-between;">
                <span>Duplikat dihapus</span>
                <b>{jumlah_duplikat:,}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    section_header(
        "Preview Dataset",
        "Sepuluh baris pertama dari data yang telah dibersihkan."
    )

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# 11. TENTANG DATASET
# ==========================================================

elif menu == "Tentang Dataset":

    page_header(
        "DATA INFORMATION",
        "Tentang Dataset",
        "Kenali struktur, atribut, dan kualitas dataset yang digunakan."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "JUMLAH DATA",
            f"{len(df):,}",
            "Baris data"
        )

    with c2:
        metric_card(
            "JUMLAH ATRIBUT",
            len(df.columns),
            "Seluruh kolom"
        )

    with c3:
        metric_card(
            "DATA DUPLIKAT",
            jumlah_duplikat,
            "Dihapus saat cleaning"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Dataset",
        "Struktur Kolom",
        "Kualitas Data"
    ])

    with tab1:

        section_header(
            "Data Bank Marketing",
            "Tampilan seluruh data setelah proses pembersihan."
        )

        st.dataframe(
            df,
            use_container_width=True,
            height=450
        )

        st.download_button(
            "Download Dataset",
            df.to_csv(index=False).encode("utf-8"),
            "bank_marketing_clean.csv",
            "text/csv"
        )

    with tab2:

        structure = pd.DataFrame({
            "Nama Kolom": df.columns,
            "Tipe Data": df.dtypes.astype(str).values,
            "Jumlah Data": df.notna().sum().values,
            "Missing Value": df.isna().sum().values
        })

        st.dataframe(
            structure,
            use_container_width=True,
            hide_index=True
        )

    with tab3:

        missing = int(df.isna().sum().sum())

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "MISSING VALUE",
                missing,
                "Nilai kosong"
            )

        with c2:
            metric_card(
                "DUPLIKAT DIHAPUS",
                jumlah_duplikat,
                "Data berulang"
            )

        with c3:
            metric_card(
                "DATA AKHIR",
                f"{len(df):,}",
                "Setelah cleaning"
            )

        quality = pd.DataFrame({
            "Pemeriksaan": [
                "Jumlah baris",
                "Jumlah kolom",
                "Missing value",
                "Duplikat dihapus"
            ],
            "Hasil": [
                len(df),
                len(df.columns),
                missing,
                jumlah_duplikat
            ]
        })

        st.dataframe(
            quality,
            use_container_width=True,
            hide_index=True
        )


# ==========================================================
# 12. EKSPLORASI DATA
# ==========================================================

elif menu == "Eksplorasi Data":

    page_header(
        "DATA EXPLORATION",
        "Eksplorasi Data",
        "Temukan pola distribusi dan karakteristik nasabah berdasarkan "
        "fitur pada dataset."
    )

    tab1, tab2, tab3 = st.tabs([
        "Target",
        "Fitur Numerik",
        "Fitur Kategorikal"
    ])

    with tab1:

        section_header(
            "Distribusi Target",
            "Jumlah nasabah berdasarkan status berlangganan deposito."
        )

        target_data = df["y"].value_counts().reset_index()
        target_data.columns = ["Status", "Jumlah"]

        fig = px.bar(
            target_data,
            x="Status",
            y="Jumlah",
            color="Status",
            text="Jumlah",
            color_discrete_map={
                "yes": "#237A62",
                "no": "#D98A45"
            }
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

        st.dataframe(
            target_data,
            use_container_width=True,
            hide_index=True
        )

    with tab2:

        section_header(
            "Analisis Fitur Numerik",
            "Distribusi nilai dan statistik deskriptif fitur numerik."
        )

        selected_num = st.selectbox(
            "Pilih fitur numerik",
            numeric_cols
        )

        fig = px.histogram(
            df,
            x=selected_num,
            color="y",
            marginal="box",
            barmode="overlay",
            opacity=0.72,
            color_discrete_map={
                "yes": "#237A62",
                "no": "#D98A45"
            }
        )

        st.plotly_chart(
            style_chart(fig, 400),
            use_container_width=True
        )

        st.markdown("#### Statistik Deskriptif")

        st.dataframe(
            df[numeric_cols].describe().T,
            use_container_width=True
        )

    with tab3:

        section_header(
            "Analisis Fitur Kategorikal",
            "Perbandingan kategori berdasarkan status berlangganan."
        )

        selected_cat = st.selectbox(
            "Pilih fitur kategorikal",
            categorical_cols
        )

        category_data = (
            df.groupby([selected_cat, "y"])
            .size()
            .reset_index(name="Jumlah")
        )

        fig = px.bar(
            category_data,
            x=selected_cat,
            y="Jumlah",
            color="y",
            barmode="group",
            color_discrete_map={
                "yes": "#237A62",
                "no": "#D98A45"
            }
        )

        st.plotly_chart(
            style_chart(fig, 400),
            use_container_width=True
        )

        st.dataframe(
            category_data,
            use_container_width=True,
            hide_index=True
        )


# ==========================================================
# 13. PREPROCESSING
# ==========================================================

elif menu == "Preprocessing":

    page_header(
        "DATA PREPARATION",
        "Preprocessing Data",
        "Persiapan data sebelum digunakan dalam proses pemodelan "
        "machine learning."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Dataset",
        "Fitur Numerik",
        "Fitur Kategorikal",
        "Ringkasan"
    ])

    with tab1:

        section_header(
            "Dataset Hasil Cleaning",
            "Data setelah pembersihan duplikat dan penyesuaian target."
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "TOTAL BARIS",
                f"{len(df):,}",
                "Dataset akhir"
            )

        with c2:
            metric_card(
                "TOTAL KOLOM",
                len(df.columns),
                "Seluruh atribut"
            )

        with c3:
            metric_card(
                "DUPLIKAT DIHAPUS",
                jumlah_duplikat,
                "Data berulang"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.dataframe(
            df,
            use_container_width=True,
            height=430
        )

        st.download_button(
            "Download Dataset Hasil Cleaning",
            df.to_csv(index=False).encode("utf-8"),
            "dataset_hasil_cleaning.csv",
            "text/csv"
        )

    with tab2:

        section_header(
            "Fitur Numerik",
            "Atribut dengan nilai angka yang dapat dihitung secara matematis."
        )

        metric_card(
            "JUMLAH FITUR NUMERIK",
            len(numeric_cols),
            "Kolom numerik yang digunakan"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.dataframe(
            df[numeric_cols],
            use_container_width=True,
            height=400
        )

        st.markdown("#### Statistik Deskriptif")

        st.dataframe(
            df[numeric_cols].describe().T,
            use_container_width=True
        )

        st.download_button(
            "Download Fitur Numerik",
            df[numeric_cols].to_csv(index=False).encode("utf-8"),
            "fitur_numerik.csv",
            "text/csv"
        )

    with tab3:

        section_header(
            "Fitur Kategorikal",
            "Atribut berbentuk kategori atau teks yang perlu diubah "
            "menjadi representasi numerik."
        )

        metric_card(
            "JUMLAH FITUR KATEGORIKAL",
            len(categorical_cols),
            "Kolom kategorikal yang digunakan"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.dataframe(
            df[categorical_cols],
            use_container_width=True,
            height=400
        )

        selected_category = st.selectbox(
            "Lihat distribusi salah satu fitur",
            categorical_cols,
            key="preprocessing_category"
        )

        category_summary = (
            df[selected_category]
            .value_counts(dropna=False)
            .reset_index()
        )

        category_summary.columns = ["Kategori", "Jumlah"]

        st.dataframe(
            category_summary,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Fitur Kategorikal",
            df[categorical_cols].to_csv(index=False).encode("utf-8"),
            "fitur_kategorikal.csv",
            "text/csv"
        )

    with tab4:

        section_header(
            "Ringkasan Tipe Data",
            "Informasi tipe data dan kondisi setiap kolom."
        )

        summary = pd.DataFrame({
            "Nama Fitur": df.columns,
            "Tipe Data": df.dtypes.astype(str).values,
            "Data Terisi": df.notna().sum().values,
            "Missing Value": df.isna().sum().values
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Pembagian Data Training dan Testing")

        c1, c2 = st.columns(2)

        with c1:
            metric_card(
                "DATA TRAINING",
                f"{len(X_train):,}",
                "80% dari dataset"
            )

        with c2:
            metric_card(
                "DATA TESTING",
                f"{len(X_test):,}",
                "20% dari dataset"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        split_df = pd.DataFrame({
            "Jenis Data": ["Training", "Testing"],
            "Jumlah Data": [len(X_train), len(X_test)],
            "Persentase": ["80%", "20%"]
        })

        st.dataframe(
            split_df,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    section_header(
        "Tahapan Preprocessing",
        "Teknik yang diterapkan sebelum data masuk ke algoritma."
    )

    process_cols = st.columns(3)

    process_items = [
        (
            "01",
            "Data Cleaning",
            "Menghapus data duplikat dan merapikan nilai target."
        ),
        (
            "02",
            "Feature Separation",
            "Memisahkan fitur numerik dan kategorikal."
        ),
        (
            "03",
            "Missing Value",
            "Median untuk numerik dan modus untuk kategorikal."
        ),
        (
            "04",
            "Encoding",
            "One-Hot Encoding untuk fitur kategorikal."
        ),
        (
            "05",
            "Scaling",
            "StandardScaler untuk fitur numerik."
        ),
        (
            "06",
            "Data Splitting",
            "Membagi data menjadi training dan testing 80:20."
        )
    ]

    for i, item in enumerate(process_items):

        with process_cols[i % 3]:

            st.markdown(f"""
                <div class="content-card" style="min-height:135px;">
                    <div style="
                        color:#237A62;
                        font-size:12px;
                        font-weight:800;">
                        STEP {item[0]}
                    </div>
                    <div style="
                        color:#203A34;
                        font-size:15px;
                        font-weight:800;
                        margin-top:7px;">
                        {item[1]}
                    </div>
                    <div style="
                        color:#71817A;
                        font-size:12px;
                        margin-top:6px;">
                        {item[2]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.info(
        "Tabel numerik dan kategorikal menampilkan data sebelum transformasi. "
        "Imputasi, encoding, dan scaling diterapkan di dalam pipeline "
        "ketika model dilatih. Kolom duration tidak digunakan sebagai "
        "fitur karena baru diketahui setelah panggilan berlangsung."
    )


# ==========================================================
# 14. PCA
# ==========================================================

elif menu == "Reduksi Dimensi (PCA)":

    page_header(
        "DIMENSIONALITY REDUCTION",
        "Reduksi Dimensi (PCA)",
        "Melihat bagaimana Principal Component Analysis mereduksi "
        "jumlah fitur dengan mempertahankan informasi data."
    )

    with st.spinner("Sedang melakukan transformasi PCA..."):

        preprocessor = create_preprocessor()
        X_processed = preprocessor.fit_transform(X)

        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()

        max_components = min(
            X_processed.shape[0],
            X_processed.shape[1]
        )

        pca_full = PCA(
            n_components=max_components,
            random_state=42
        )

        pca_full.fit(X_processed)

        cumulative = np.cumsum(
            pca_full.explained_variance_ratio_
        )

        component_count = np.searchsorted(
            cumulative,
            0.95
        ) + 1

        component_count = min(
            component_count,
            max_components
        )

        pca = PCA(
            n_components=component_count,
            random_state=42
        )

        X_pca = pca.fit_transform(X_processed)

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "FITUR SEBELUM PCA",
            X_processed.shape[1],
            "Dimensi awal"
        )

    with c2:
        metric_card(
            "KOMPONEN PCA",
            component_count,
            "Komponen terpilih"
        )

    with c3:
        metric_card(
            "VARIANSI",
            f"{sum(pca.explained_variance_ratio_) * 100:.2f}%",
            "Informasi yang dipertahankan"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    section_header(
        "Cumulative Explained Variance",
        "Menunjukkan akumulasi informasi yang dapat dipertahankan "
        "oleh setiap komponen utama."
    )

    variance_df = pd.DataFrame({
        "Komponen": range(1, len(cumulative) + 1),
        "Variansi Kumulatif": cumulative
    })

    fig = px.line(
        variance_df,
        x="Komponen",
        y="Variansi Kumulatif",
        markers=True
    )

    fig.add_hline(
        y=0.95,
        line_dash="dash",
        line_color="#D98A45",
        annotation_text="Target 95%"
    )

    st.plotly_chart(
        style_chart(fig, 400),
        use_container_width=True
    )

    section_header(
        "Hasil Transformasi",
        "Contoh data setelah direpresentasikan dalam komponen utama."
    )

    pca_df = pd.DataFrame(
        X_pca[:100],
        columns=[
            f"PC{i+1}" for i in range(component_count)
        ]
    )

    pca_df["Target"] = y.iloc[:100].values

    st.dataframe(
        pca_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "PCA ditampilkan sebagai analisis reduksi dimensi. "
        "Model klasifikasi pada halaman evaluasi menggunakan pipeline "
        "fitur tersendiri, bukan hasil PCA."
    )


# ==========================================================
# 15. MODELLING & EVALUASI
# ==========================================================

elif menu == "Modelling & Evaluasi":

    page_header(
        "MODEL PERFORMANCE",
        "Modelling & Evaluasi",
        "Membandingkan performa lima algoritma klasifikasi "
        "untuk menentukan model terbaik."
    )

    with st.spinner("Melatih model dan menghitung evaluasi..."):
        trained_models, results_df, best_name = train_models()

    best_result = results_df.iloc[0]

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "MODEL TERBAIK",
            best_name,
            "Berdasarkan F1 Score"
        )

    with c2:
        metric_card(
            "ACCURACY",
            format_percent(best_result["Accuracy"]),
            "Model terbaik"
        )

    with c3:
        metric_card(
            "F1 SCORE",
            format_percent(best_result["F1 Score"]),
            "Model terbaik"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    section_header(
        "Perbandingan Model",
        "Hasil evaluasi dari setiap algoritma yang diuji."
    )

    display_results = results_df.copy()

    for col in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]:
        display_results[col] = display_results[col].map(
            lambda x: f"{x:.4f}"
        )

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    selected_metric = st.selectbox(
        "Pilih metrik untuk dibandingkan",
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    )

    fig = px.bar(
        results_df,
        x="Model",
        y=selected_metric,
        color="Model",
        text_auto=".3f",
        color_discrete_sequence=[
            "#237A62",
            "#5C8D7A",
            "#D98A45",
            "#7896A2",
            "#9B8B75"
        ]
    )

    fig.update_layout(showlegend=False)

    st.plotly_chart(
        style_chart(fig, 400),
        use_container_width=True
    )

    st.divider()

    best_model = trained_models[best_name]

    y_pred = best_model.predict(X_test)

    y_prob = best_model.predict_proba(X_test)[:, 1]

    section_header(
        "Evaluasi Model Terbaik",
        f"Detail evaluasi untuk algoritma {best_name}."
    )

    left, right = st.columns(2)

    with left:

        st.markdown("#### Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Prediksi No", "Prediksi Yes"],
            y=["Aktual No", "Aktual Yes"],
            color_continuous_scale=[
                "#F0F5F2",
                "#237A62"
            ],
            aspect="auto"
        )

        st.plotly_chart(
            style_chart(fig, 350),
            use_container_width=True
        )

    with right:

        st.markdown("#### Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            target_names=[
                "Tidak Berlangganan",
                "Berlangganan"
            ],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).T

        st.dataframe(
            report_df,
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    section_header(
        "ROC Curve",
        "Menggambarkan kemampuan model membedakan dua kelas."
    )

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name="ROC Curve",
        line=dict(color="#237A62", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(color="#D98A45", dash="dash")
    ))

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(
        style_chart(fig, 400),
        use_container_width=True
    )

    st.success(
        f"Model terbaik berdasarkan F1 Score adalah {best_name}."
    )


# ==========================================================
# 16. PREDIKSI SATU NASABAH
# ==========================================================

elif menu == "Prediksi Satu Nasabah":

    page_header(
        "CUSTOMER PREDICTION",
        "Prediksi Satu Nasabah",
        "Masukkan karakteristik nasabah untuk memperoleh estimasi "
        "potensi berlangganan deposito."
    )

    with st.spinner("Menyiapkan model prediksi..."):
        trained_models, results_df, best_name = train_models()

    best_model = trained_models[best_name]
    best_result = results_df.iloc[0]

    st.markdown(f"""
        <div class="content-card" style="
            display:flex;
            justify-content:space-between;
            align-items:center;">
            <div>
                <div style="font-size:11px;color:#71817A;">
                    MODEL PREDIKSI AKTIF
                </div>
                <div style="
                    font-size:17px;
                    font-weight:800;
                    color:#172B2A;
                    margin-top:4px;">
                    {best_name}
                </div>
            </div>
            <div style="
                background:#E8F3EE;
                color:#237A62;
                padding:7px 12px;
                border-radius:6px;
                font-size:11px;
                font-weight:800;">
                ACTIVE MODEL
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    input_values = {}

    with st.form("customer_prediction_form"):

        section_header(
            "Informasi Nasabah",
            "Isi data berdasarkan informasi nasabah yang akan diprediksi."
        )

        col1, col2 = st.columns(2)

        for i, feature in enumerate(FEATURES):

            target_col = col1 if i % 2 == 0 else col2

            with target_col:

                label = feature.replace("_", " ").title()

                if feature in numeric_cols:

                    median_value = df[feature].median()

                    if pd.isna(median_value):
                        median_value = 0

                    input_values[feature] = st.number_input(
                        label,
                        value=float(median_value),
                        key=f"input_{feature}"
                    )

                else:

                    options = sorted(
                        df[feature]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    if options:

                        input_values[feature] = st.selectbox(
                            label,
                            options,
                            key=f"input_{feature}"
                        )

                    else:
                        input_values[feature] = ""

        submitted = st.form_submit_button(
            "Jalankan Prediksi",
            type="primary",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame(
            [input_values],
            columns=FEATURES
        )

        # --------------------------------------------------
        # PROSES PREDIKSI
        # --------------------------------------------------

        prediction, probability_yes, probability_no = (
            get_prediction_details(best_model, input_df)
        )

        probability_yes_percent = probability_yes * 100
        probability_no_percent = probability_no * 100

        st.divider()

        section_header(
            "Hasil Prediksi",
            "Hasil klasifikasi berdasarkan model machine learning."
        )

        if prediction == 1:

            st.markdown("""
                <div class="result-positive">
                    <div style="
                        color:#237A62;
                        font-size:11px;
                        font-weight:800;
                        letter-spacing:1px;">
                        HASIL KLASIFIKASI
                    </div>
                    <div class="result-title" style="color:#1D684F;">
                        Nasabah Berpotensi Berlangganan Deposito
                    </div>
                    <div class="result-description">
                        Berdasarkan pola yang dipelajari model,
                        karakteristik nasabah ini lebih mengarah pada
                        kemungkinan berlangganan deposito.
                    </div>
                </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
                <div class="result-negative">
                    <div style="
                        color:#B6534D;
                        font-size:11px;
                        font-weight:800;
                        letter-spacing:1px;">
                        HASIL KLASIFIKASI
                    </div>
                    <div class="result-title" style="color:#A8433D;">
                        Nasabah Diprediksi Tidak Berlangganan
                    </div>
                    <div class="result-description">
                        Berdasarkan pola yang dipelajari model,
                        karakteristik nasabah ini lebih mengarah pada
                        kemungkinan tidak berlangganan deposito.
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            metric_card(
                "PROBABILITAS BERLANGGANAN",
                f"{probability_yes_percent:.2f}%",
                "Kelas Yes"
            )

        with c2:
            metric_card(
                "PROBABILITAS TIDAK BERLANGGANAN",
                f"{probability_no_percent:.2f}%",
                "Kelas No"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # --------------------------------------------------
        # VISUALISASI PROBABILITAS
        # --------------------------------------------------

        section_header(
            "Perbandingan Probabilitas",
            "Perbandingan nilai probabilitas dari kedua kelas."
        )

        probability_df = pd.DataFrame({
            "Status": [
                "Berlangganan",
                "Tidak Berlangganan"
            ],
            "Probabilitas": [
                probability_yes_percent,
                probability_no_percent
            ]
        })

        fig = px.bar(
            probability_df,
            x="Status",
            y="Probabilitas",
            color="Status",
            text_auto=".2f",
            color_discrete_map={
                "Berlangganan": "#237A62",
                "Tidak Berlangganan": "#D98A45"
            }
        )

        fig.update_layout(showlegend=False)

        fig.update_yaxes(
            title="Probabilitas (%)",
            range=[0, 100]
        )

        st.plotly_chart(
            style_chart(fig, 330),
            use_container_width=True
        )

        # --------------------------------------------------
        # PENJELASAN CARA KERJA MODEL
        # --------------------------------------------------

        st.divider()

        section_header(
            "Bagaimana Model Menghasilkan Prediksi?",
            "Alur pengolahan data dari input nasabah sampai menghasilkan "
            "keputusan klasifikasi."
        )

        step1, step2, step3 = st.columns(3)

        with step1:
            st.markdown("""
                <div class="calculation-card">
                    <div class="calculation-number">TAHAP 01</div>
                    <div class="calculation-title">Input Data</div>
                    <div class="calculation-description">
                        Data nasabah dimasukkan berdasarkan atribut
                        seperti pekerjaan, usia, status perkawinan,
                        pendidikan, dan informasi lainnya.
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with step2:
            st.markdown("""
                <div class="calculation-card">
                    <div class="calculation-number">TAHAP 02</div>
                    <div class="calculation-title">Preprocessing</div>
                    <div class="calculation-description">
                        Data numerik melalui proses imputasi dan
                        StandardScaler. Data kategorikal diubah
                        menggunakan One-Hot Encoding.
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with step3:
            st.markdown("""
                <div class="calculation-card">
                    <div class="calculation-number">TAHAP 03</div>
                    <div class="calculation-title">Klasifikasi</div>
                    <div class="calculation-description">
                        Model membandingkan pola input dengan pola
                        yang dipelajari dari data training, kemudian
                        menghasilkan prediksi dan probabilitas.
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        section_header(
            "Perhitungan Probabilitas",
            "Model menghasilkan nilai probabilitas untuk masing-masing kelas."
        )

        calculation_left, calculation_right = st.columns(2)

        with calculation_left:

            st.markdown("""
                <div class="content-card">
                    <div style="
                        color:#237A62;
                        font-size:11px;
                        font-weight:800;">
                        KELAS YES
                    </div>
                    <div style="
                        color:#203A34;
                        font-size:23px;
                        font-weight:800;
                        margin-top:8px;">
                        Probabilitas Berlangganan
                    </div>
                    <div style="
                        color:#71817A;
                        font-size:13px;
                        margin-top:8px;">
                        Menggambarkan tingkat keyakinan model
                        terhadap kemungkinan nasabah berlangganan deposito.
                    </div>
                    <hr>
                    <div style="font-size:13px;color:#53665F;">
                        Nilai probabilitas:
                    </div>
                    <div style="
                        font-size:25px;
                        color:#237A62;
                        font-weight:800;
                        margin-top:5px;">
                        {:.2f}%
                    </div>
                </div>
            """.format(probability_yes_percent), unsafe_allow_html=True)

        with calculation_right:

            st.markdown("""
                <div class="content-card">
                    <div style="
                        color:#D98A45;
                        font-size:11px;
                        font-weight:800;">
                        KELAS NO
                    </div>
                    <div style="
                        color:#203A34;
                        font-size:23px;
                        font-weight:800;
                        margin-top:8px;">
                        Probabilitas Tidak Berlangganan
                    </div>
                    <div style="
                        color:#71817A;
                        font-size:13px;
                        margin-top:8px;">
                        Menggambarkan tingkat keyakinan model
                        terhadap kemungkinan nasabah tidak berlangganan.
                    </div>
                    <hr>
                    <div style="font-size:13px;color:#53665F;">
                        Nilai probabilitas:
                    </div>
                    <div style="
                        font-size:25px;
                        color:#D98A45;
                        font-weight:800;
                        margin-top:5px;">
                        {:.2f}%
                    </div>
                </div>
            """.format(probability_no_percent), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
            <div class="content-card">
                <div style="
                    font-size:14px;
                    font-weight:800;
                    color:#203A34;">
                    Aturan Penentuan Hasil
                </div>
                <p style="font-size:13px;margin-top:8px;">
                    Model membandingkan probabilitas dari dua kelas.
                    Kelas dengan probabilitas lebih tinggi menjadi
                    hasil prediksi akhir.
                </p>
                <div style="
                    background:#F3F6F5;
                    padding:13px;
                    border-radius:8px;
                    font-size:13px;
                    color:#203A34;
                    font-weight:700;">
                    Jika P(Yes) lebih besar dari P(No), maka hasil = Yes.
                    <br>
                    Jika P(No) lebih besar atau sama dengan P(Yes),
                    maka hasil = No.
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.info(
            "Catatan: angka probabilitas merupakan estimasi yang dihasilkan "
            "oleh model berdasarkan pola data training. Nilai tersebut "
            "bukan jaminan bahwa nasabah benar-benar akan berlangganan."
        )

        # --------------------------------------------------
        # DATA NASABAH
        # --------------------------------------------------

        st.divider()

        section_header(
            "Data Nasabah",
            "Data yang dimasukkan ke dalam model."
        )

        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # DATA PENDUKUNG
        # --------------------------------------------------

        st.divider()

        section_header(
            "Data Pendukung",
            "Perbandingan karakteristik nasabah dengan kelompok "
            "nasabah pada dataset."
        )

        support_rows = []

        for feature in FEATURES:

            value = input_values[feature]

            if feature in numeric_cols:

                yes_mean = df.loc[
                    df["y"] == "yes",
                    feature
                ].mean()

                no_mean = df.loc[
                    df["y"] == "no",
                    feature
                ].mean()

                support_rows.append({
                    "Fitur": feature,
                    "Nilai Nasabah": value,
                    "Rata-rata Kelompok Yes": round(yes_mean, 2),
                    "Rata-rata Kelompok No": round(no_mean, 2)
                })

            else:

                yes_percentage = (
                    (
                        df.loc[df["y"] == "yes", feature]
                        .astype(str) == str(value)
                    ).mean() * 100
                )

                no_percentage = (
                    (
                        df.loc[df["y"] == "no", feature]
                        .astype(str) == str(value)
                    ).mean() * 100
                )

                support_rows.append({
                    "Fitur": feature,
                    "Nilai Nasabah": value,
                    "Persentase Kelompok Yes": round(yes_percentage, 2),
                    "Persentase Kelompok No": round(no_percentage, 2)
                })

        support_df = pd.DataFrame(support_rows)

        st.dataframe(
            support_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Data pendukung merupakan perbandingan deskriptif "
            "dari dataset, bukan ukuran kontribusi langsung "
            "setiap fitur terhadap hasil prediksi."
        )

        # --------------------------------------------------
        # PERFORMA MODEL
        # --------------------------------------------------

        st.divider()

        section_header(
            "Informasi Performa Model",
            "Metrik evaluasi model yang digunakan untuk prediksi."
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "MODEL",
                best_name,
                "Algoritma terpilih"
            )

        with c2:
            metric_card(
                "ACCURACY",
                format_percent(best_result["Accuracy"]),
                "Data testing"
            )

        with c3:
            metric_card(
                "F1 SCORE",
                format_percent(best_result["F1 Score"]),
                "Data testing"
            )


# ==========================================================
# 17. PREDIKSI MASSAL
# ==========================================================

elif menu == "Prediksi Massal (CSV)":

    page_header(
        "BATCH PREDICTION",
        "Prediksi Massal",
        "Prediksi banyak nasabah sekaligus menggunakan file CSV."
    )

    with st.spinner("Menyiapkan model prediksi..."):
        trained_models, results_df, best_name = train_models()

    best_model = trained_models[best_name]

    st.markdown(f"""
        <div class="content-card">
            <div style="font-size:11px;color:#71817A;">
                MODEL YANG DIGUNAKAN
            </div>
            <div style="
                font-size:18px;
                color:#172B2A;
                font-weight:800;
                margin-top:5px;">
                {best_name}
            </div>
        </div>
    """, unsafe_allow_html=True)

    section_header(
        "Langkah Pertama: Download Template",
        "Gunakan template agar nama dan susunan kolom sesuai dengan "
        "fitur yang dibutuhkan model."
    )

    template = pd.DataFrame(columns=FEATURES)

    for feature in FEATURES:

        if feature in numeric_cols:
            template.loc[0, feature] = df[feature].median()

        else:
            values = df[feature].dropna()
            template.loc[0, feature] = (
                values.iloc[0] if len(values) else ""
            )

    st.download_button(
        "Download Template CSV",
        template.to_csv(index=False).encode("utf-8"),
        "template_prediksi_nasabah.csv",
        "text/csv"
    )

    st.divider()

    section_header(
        "Langkah Kedua: Upload Dataset",
        "Unggah file CSV yang berisi data nasabah."
    )

    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            upload_df = pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python"
            )

            upload_df.columns = upload_df.columns.str.strip()

            if TARGET in upload_df.columns:
                upload_df = upload_df.drop(columns=[TARGET])

            missing_features = [
                feature for feature in FEATURES
                if feature not in upload_df.columns
            ]

            if missing_features:

                st.error(
                    "Kolom belum lengkap: "
                    + ", ".join(missing_features)
                )

            else:

                prediction_input = upload_df[FEATURES].copy()

                predictions = best_model.predict(
                    prediction_input
                )

                probabilities = best_model.predict_proba(
                    prediction_input
                )

                model_classes = list(best_model.classes_)
                positive_index = model_classes.index(1)

                positive_probabilities = probabilities[:, positive_index]

                result_df = upload_df.copy()

                result_df["Hasil Prediksi"] = np.where(
                    predictions == 1,
                    "Berlangganan",
                    "Tidak Berlangganan"
                )

                result_df["Probabilitas Berlangganan (%)"] = (
                    positive_probabilities * 100
                ).round(2)

                st.divider()

                section_header(
                    "Ringkasan Hasil Prediksi",
                    "Jumlah nasabah berdasarkan hasil klasifikasi."
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    metric_card(
                        "TOTAL NASABAH",
                        f"{len(result_df):,}",
                        "Data yang diprediksi"
                    )

                with c2:
                    metric_card(
                        "BERLANGGANAN",
                        f"{int((predictions == 1).sum()):,}",
                        "Prediksi Yes"
                    )

                with c3:
                    metric_card(
                        "TIDAK BERLANGGANAN",
                        f"{int((predictions == 0).sum()):,}",
                        "Prediksi No"
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                section_header(
                    "Tabel Hasil Prediksi",
                    "Hasil klasifikasi seluruh nasabah dari file yang diunggah."
                )

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    height=400
                )

                st.download_button(
                    "Download Hasil Prediksi",
                    result_df.to_csv(index=False).encode("utf-8"),
                    "hasil_prediksi_nasabah.csv",
                    "text/csv"
                )

                section_header(
                    "Distribusi Hasil",
                    "Proporsi hasil klasifikasi dari seluruh nasabah."
                )

                fig = px.pie(
                    result_df,
                    names="Hasil Prediksi",
                    hole=0.6,
                    color="Hasil Prediksi",
                    color_discrete_map={
                        "Berlangganan": "#237A62",
                        "Tidak Berlangganan": "#D98A45"
                    }
                )

                fig.update_traces(
                    textposition="outside",
                    textinfo="label+percent"
                )

                st.plotly_chart(
                    style_chart(fig, 380),
                    use_container_width=True
                )

        except Exception as error:
            st.error(f"Gagal memproses file: {error}")


# ==========================================================
# 18. KESIMPULAN & SARAN
# ==========================================================

elif menu == "Kesimpulan & Saran":

    page_header(
        "PROJECT SUMMARY",
        "Kesimpulan & Saran",
        "Ringkasan hasil analisis, performa model, dan rekomendasi "
        "pengembangan selanjutnya."
    )

    with st.spinner("Mengambil hasil evaluasi model..."):
        trained_models, results_df, best_name = train_models()

    best_result = results_df.iloc[0]

    section_header(
        "Kesimpulan Analisis",
        "Ringkasan hasil dari seluruh tahapan penelitian."
    )

    st.markdown(f"""
        <div class="content-card">
            <p>
                Dataset Bank Marketing memiliki <b>{len(df):,} data nasabah</b>
                setelah proses pembersihan. Data tersebut digunakan untuk
                membangun model klasifikasi yang memprediksi kemungkinan
                nasabah berlangganan deposito berjangka.
            </p>
            <p>
                Lima algoritma dibandingkan, yaitu Logistic Regression,
                Decision Tree, Random Forest, SVM, dan Gradient Boosting.
                Berdasarkan nilai F1 Score, model terbaik adalah:
            </p>
            <div style="
                background:#E8F3EE;
                border:1px solid #C7DED2;
                padding:15px;
                border-radius:8px;
                margin-top:12px;">
                <div style="
                    font-size:11px;
                    color:#527767;
                    font-weight:800;">
                    MODEL TERBAIK
                </div>
                <div style="
                    font-size:22px;
                    font-weight:800;
                    color:#1D684F;
                    margin-top:5px;">
                    {best_name}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "ACCURACY",
            format_percent(best_result["Accuracy"]),
            "Ketepatan prediksi"
        )

    with c2:
        metric_card(
            "PRECISION",
            format_percent(best_result["Precision"]),
            "Ketepatan prediksi positif"
        )

    with c3:
        metric_card(
            "F1 SCORE",
            format_percent(best_result["F1 Score"]),
            "Keseimbangan precision dan recall"
        )

    st.divider()

    section_header(
        "Saran Pengembangan",
        "Beberapa hal yang dapat dilakukan untuk meningkatkan kualitas sistem."
    )

    suggestions = [
        (
            "Optimasi Model",
            "Melakukan hyperparameter tuning untuk memperoleh konfigurasi "
            "model yang lebih optimal."
        ),
        (
            "Penambahan Data",
            "Menggunakan data nasabah yang lebih beragam agar model "
            "mempelajari pola yang lebih luas."
        ),
        (
            "Interpretasi Model",
            "Menambahkan metode interpretasi seperti SHAP untuk memahami "
            "kontribusi fitur terhadap prediksi."
        ),
        (
            "Evaluasi Berkala",
            "Melakukan evaluasi ulang ketika tersedia data terbaru "
            "agar performa model tetap terpantau."
        )
    ]

    for title, description in suggestions:

        st.markdown(f"""
            <div class="content-card" style="
                display:flex;
                gap:15px;
                align-items:flex-start;">
                <div style="
                    background:#E8F3EE;
                    color:#237A62;
                    font-weight:800;
                    border-radius:8px;
                    padding:8px 11px;">
                    →
                </div>
                <div>
                    <div style="
                        font-weight:800;
                        color:#203A34;
                        font-size:14px;">
                        {title}
                    </div>
                    <div style="
                        font-size:12px;
                        color:#71817A;
                        margin-top:4px;">
                        {description}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    section_header(
        "Perbandingan Akhir Model",
        "Seluruh metrik evaluasi dari model yang diuji."
    )

    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "ROC-AUC": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# 19. FOOTER
# ==========================================================

st.markdown("""
    <div class="footer">
        BANK MARKETING ANALYTICS &nbsp;·&nbsp;
        MACHINE LEARNING CLASSIFICATION
    </div>
""", unsafe_allow_html=True)