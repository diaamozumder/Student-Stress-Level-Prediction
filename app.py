
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Student Stress Level Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# PROFESSIONAL UI / CSS
# =========================================================
st.markdown(
    """
    <style>
        /* Main background */
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37,99,235,0.08), transparent 35%),
                radial-gradient(circle at top right, rgba(14,165,233,0.08), transparent 30%),
                #f8fafc;
        }

        /* Reduce excess top spacing */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1250px;
        }

        /* Hero */
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #0369a1 100%);
            border-radius: 24px;
            padding: 34px 36px;
            color: white;
            margin-bottom: 24px;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.16);
        }

        .hero-title {
            font-size: 42px;
            line-height: 1.1;
            font-weight: 800;
            margin: 0 0 8px 0;
            letter-spacing: -0.8px;
        }

        .hero-subtitle {
            font-size: 16px;
            color: rgba(255,255,255,0.82);
            margin: 0;
            max-width: 780px;
        }

        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 999px;
            padding: 6px 11px;
            font-size: 12px;
            margin-bottom: 14px;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.92);
            border: 1px solid #e2e8f0;
            padding: 18px 18px 15px 18px;
            border-radius: 16px;
            box-shadow: 0 8px 22px rgba(15,23,42,0.05);
        }

        div[data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 800;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            font-weight: 700;
        }

        /* Headings / labels */
        h1, h2, h3 {
            color: #0f172a;
        }

        label, .stCaption, p {
            color: #334155;
        }

        /* Form input appearance */
        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {
            border-radius: 10px !important;
        }

        div[data-baseweb="select"] > div {
            border-color: #cbd5e1 !important;
        }

        /* Prediction card */
        .prediction-card {
            border-radius: 20px;
            padding: 28px;
            margin-top: 18px;
            text-align: center;
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 12px 28px rgba(15,23,42,0.08);
        }

        .prediction-low {
            background: linear-gradient(135deg, #ecfdf5, #dcfce7);
            border-left: 7px solid #16a34a;
        }

        .prediction-moderate {
            background: linear-gradient(135deg, #fffbeb, #fef3c7);
            border-left: 7px solid #d97706;
        }

        .prediction-high {
            background: linear-gradient(135deg, #fff1f2, #fee2e2);
            border-left: 7px solid #dc2626;
        }

        .prediction-label {
            color: #64748b;
            font-size: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .prediction-value {
            color: #0f172a;
            font-size: 36px;
            font-weight: 900;
            margin: 8px 0;
        }

        .prediction-model {
            color: #475569;
            font-size: 14px;
        }

        /* Small information card */
        .soft-card {
            background: rgba(255,255,255,0.88);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 22px rgba(15,23,42,0.04);
            margin: 8px 0 18px 0;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #94a3b8;
            font-size: 13px;
            padding-top: 14px;
        }

        /* Hide Streamlit menu/footer for cleaner demo */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        @media (max-width: 768px) {
            .hero-title {font-size: 30px;}
            .hero {padding: 26px 22px;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
TARGET = "Overall, how would you describe your current stress level?"
DATA_FILE = "student_stress_cleaned.csv"


def friendly_label(text):
    """Make long Google Form column names cleaner for the UI."""
    text = str(text).replace("\n", " ").strip()
    text = " ".join(text.split())
    return text


def feature_group(feature):
    """UI-only grouping. Does not change the ML model or data."""
    name = feature.lower()

    academic_words = [
        "department", "academic year", "cg", "study", "class",
        "exam", "quiz", "academic", "concentrat", "career", "time manage"
    ]
    lifestyle_words = [
        "sleep", "exercise", "social media", "part-time", "tired",
        "health", "free time"
    ]

    if any(word in name for word in academic_words):
        return "Academic Factors"
    if any(word in name for word in lifestyle_words):
        return "Lifestyle Factors"
    return "Well-being & Personal Factors"


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    data.columns = data.columns.astype(str).str.strip()

    # Remove accidental CSV index columns if present
    unnamed = [c for c in data.columns if c.lower().startswith("unnamed:")]
    if unnamed:
        data = data.drop(columns=unnamed)

    return data


# =========================================================
# LOAD + VALIDATE DATA
# =========================================================
try:
    df = load_data()
except FileNotFoundError:
    st.error(
        f"Could not find `{DATA_FILE}`. Put it in the same folder as `app.py`."
    )
    st.stop()

if TARGET not in df.columns:
    st.error("The target stress-level column was not found in the CSV.")
    st.code(TARGET)
    st.stop()

# Remove rows where the target is missing
df = df.dropna(subset=[TARGET]).copy()

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(str)

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]


# =========================================================
# TARGET ENCODING
# =========================================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded,
)


# =========================================================
# PREPROCESSING
# =========================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
    ],
    remainder="drop",
)


# =========================================================
# MODELS
# =========================================================
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
    ),
    "SVM": SVC(
        class_weight="balanced",
        probability=True,
        random_state=42,
    ),
}


# =========================================================
# TRAIN MODELS
# =========================================================
@st.cache_resource
def train_models():
    trained = {}
    scores = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        trained[name] = pipeline
        scores[name] = accuracy_score(y_test, predictions)

    return trained, scores


trained_models, results = train_models()

best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]


# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="badge">🎓 MACHINE LEARNING PROJECT</div>
        <div class="hero-title">Student Stress Level Prediction</div>
        <p class="hero-subtitle">
            A simple machine-learning web application that predicts a student's
            reported stress level as Low, Moderate, or High using academic,
            lifestyle, and self-reported well-being factors.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SUMMARY METRICS
# =========================================================
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Usable Records", len(df))

with m2:
    st.metric("Input Features", X.shape[1])

with m3:
    st.metric("Models Compared", len(models))

with m4:
    st.metric("Stress Classes", len(label_encoder.classes_))


# =========================================================
# TABS
# =========================================================
tab_predict, tab_models, tab_about = st.tabs(
    [
        "🔮 Predict Stress",
        "📊 Model Comparison",
        "ℹ️ Project Overview",
    ]
)


# =========================================================
# TAB 1 — PREDICTION
# =========================================================
with tab_predict:
    left, right = st.columns([2.2, 1])

    with left:
        st.subheader("Student Information")
        st.caption(
            "Fill in the information below. All fields are based on the features used by the trained model."
        )

    with right:
        selected_model_name = st.selectbox(
            "Prediction model",
            list(trained_models.keys()),
            index=list(trained_models.keys()).index(best_model_name),
            help="You can demonstrate predictions using any of the five trained models.",
        )

    user_input = {}

    with st.form("prediction_form"):
        groups = [
            "Academic Factors",
            "Lifestyle Factors",
            "Well-being & Personal Factors",
        ]

        for group in groups:
            group_features = [
                feature for feature in X.columns
                if feature_group(feature) == group
            ]

            if not group_features:
                continue

            st.markdown(f"### {group}")
            c1, c2 = st.columns(2)

            for index, feature in enumerate(group_features):
                container = c1 if index % 2 == 0 else c2
                label = friendly_label(feature)

                with container:
                    if feature in numeric_features:
                        series = pd.to_numeric(X[feature], errors="coerce").dropna()

                        if series.empty:
                            user_input[feature] = st.number_input(
                                label,
                                value=0.0,
                                key=f"input_{feature}",
                            )
                            continue

                        minimum = float(series.min())
                        maximum = float(series.max())
                        default = float(series.median())

                        # FIXED:
                        # Works for Python int, NumPy int, and float values.
                        is_integer_like = bool(
                            np.all(np.isclose(series.to_numpy(dtype=float),
                                              np.round(series.to_numpy(dtype=float))))
                        )

                        if is_integer_like:
                            user_input[feature] = st.number_input(
                                label,
                                min_value=int(np.floor(minimum)),
                                max_value=int(np.ceil(maximum)),
                                value=int(round(default)),
                                step=1,
                                key=f"input_{feature}",
                            )
                        else:
                            user_input[feature] = st.number_input(
                                label,
                                min_value=minimum,
                                max_value=maximum,
                                value=default,
                                step=0.01,
                                format="%.2f",
                                key=f"input_{feature}",
                            )

                    else:
                        options = (
                            X[feature]
                            .dropna()
                            .astype(str)
                            .drop_duplicates()
                            .tolist()
                        )

                        if not options:
                            options = ["Unknown"]

                        user_input[feature] = st.selectbox(
                            label,
                            options=options,
                            key=f"input_{feature}",
                        )

            st.divider()

        submitted = st.form_submit_button(
            "✨ Predict Stress Level",
            use_container_width=True,
        )

    if submitted:
        input_df = pd.DataFrame([user_input], columns=X.columns)

        selected_model = trained_models[selected_model_name]
        encoded_prediction = selected_model.predict(input_df)[0]
        prediction = label_encoder.inverse_transform(
            [int(encoded_prediction)]
        )[0]

        prediction_clean = str(prediction).strip().lower()

        if prediction_clean == "low":
            card_class = "prediction-low"
            icon = "🟢"
        elif prediction_clean == "moderate":
            card_class = "prediction-moderate"
            icon = "🟡"
        else:
            card_class = "prediction-high"
            icon = "🔴"

        st.markdown(
            f"""
            <div class="prediction-card {card_class}">
                <div class="prediction-label">Predicted Stress Level</div>
                <div class="prediction-value">{icon} {str(prediction).upper()}</div>
                <div class="prediction-model">
                    Model used: <strong>{selected_model_name}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "Academic demonstration only — this prediction is not a medical or clinical diagnosis."
        )


# =========================================================
# TAB 2 — MODEL COMPARISON
# =========================================================
with tab_models:
    st.subheader("Model Performance Comparison")
    st.caption(
        "All models use the same train/test split so their test accuracies can be compared fairly within this application."
    )

    result_df = pd.DataFrame(
        {
            "Model": list(results.keys()),
            "Accuracy (%)": [
                round(score * 100, 2)
                for score in results.values()
            ],
        }
    ).sort_values("Accuracy (%)", ascending=False)

    c1, c2 = st.columns([1.05, 1.4])

    with c1:
        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            f"""
            <div class="soft-card">
                <strong>Highest accuracy in this app:</strong><br>
                {best_model_name} — {best_accuracy * 100:.2f}%
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        chart_df = result_df.set_index("Model")
        st.bar_chart(chart_df["Accuracy (%)"])

    st.markdown("### Models Incorporated")
    model_cols = st.columns(5)
    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "SVM",
    ]
    icons = ["📈", "🌳", "🌲", "👥", "📐"]

    for col, icon, name in zip(model_cols, icons, model_names):
        with col:
            st.markdown(
                f"""
                <div class="soft-card" style="text-align:center; min-height:110px;">
                    <div style="font-size:28px;">{icon}</div>
                    <strong>{name}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Regularisation / Complexity Control")
    st.write(
        "The project includes Logistic Regression and SVM, which use regularisation through the "
        "`C` parameter. Your notebook also contains the L1 vs L2 regularisation comparison."
    )


# =========================================================
# TAB 3 — PROJECT OVERVIEW
# =========================================================
with tab_about:
    st.subheader("Project Overview")

    a1, a2 = st.columns(2)

    with a1:
        st.markdown(
            """
            <div class="soft-card">
                <h3>🎯 Objective</h3>
                <p>
                    Predict a student's current stress level as
                    <strong>Low, Moderate, or High</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="soft-card">
                <h3>🗂️ Data Source</h3>
                <p>
                    Anonymous student survey data collected using Google Forms.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:
        st.markdown(
            """
            <div class="soft-card">
                <h3>⚙️ ML Workflow</h3>
                <p>
                    Survey → Cleaning → EDA → Encoding & Scaling →
                    Train/Test Split → Multiple Models → Evaluation → Prediction
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="soft-card">
                <h3>🧠 Models</h3>
                <p>
                    Logistic Regression, Decision Tree, Random Forest,
                    K-Nearest Neighbors, and Support Vector Machine.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.warning(
        "This system is an academic machine-learning demonstration and should not be used as a clinical mental-health assessment."
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        Student Stress Level Prediction • Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True,
)
