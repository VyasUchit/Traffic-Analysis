import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Smart Traffic Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .main {
        background-color: #f5f8fc;
    }
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }

    h1, h2, h3, p, label {
        color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(120deg, #102b50, #1764a5);
        padding: 35px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: white;
        font-size: 36px;
    }

    .hero p {
        color: #e5efff;
        font-size: 17px;
    }

    .section-title {
        color: #15345d;
        font-size: 25px;
        font-weight: 700;
    }
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        padding: 18px;
        border: 1px solid #334155;
        border-radius: 12px;
        color: white;
    }

    div[data-testid="stMetric"] label {
        color: #cbd5e1 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #38bdf8 !important;
    }

    .result-card {
        padding: 20px;
        background: #e9f7ef;
        border: 1px solid #b8e5c8;
        border-radius: 12px;
        margin-top: 15px;
    }

    .footer {
        text-align: center;
        color: gray;
        padding: 20px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 15px 5px;">
        <div style="
            background-color: #2563eb;
            width: 75px;
            height: 75px;
            border-radius: 50%;
            margin: auto;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 30px;
            font-weight: bold;
            color: white;
        ">
            UV
        </div>
        <h3 style="margin-bottom: 2px; color: white;">
            Uchit Vyas
        </h3>
        <p style="font-size: 13px; color: #94a3b8;">
            Machine Learning Developer
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

st.sidebar.title("🚦 Smart Traffic")
st.sidebar.caption("Traffic Analytics System")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Traffic Prediction", "Analytics"],
    index=0
)

st.sidebar.divider()

st.sidebar.subheader("About Me")
st.sidebar.write("B.Tech IT Student")
st.sidebar.write("AI & Data Science")
st.sidebar.write("Python | Machine Learning")

st.sidebar.divider()

st.sidebar.caption("Developed by Uchit Vyas")

@st.cache_resource
def load_model():
    model = joblib.load("traffic_volume_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor


@st.cache_data
def load_dataset():
    df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date_time", "traffic_volume"]
    ).copy()

    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month

    return df


try:
    model, preprocessor = load_model()
    model_error = None
except Exception as e:
    model = None
    preprocessor = None
    model_error = str(e)

try:
    df = load_dataset()
    data_error = None
except Exception as e:
    df = None
    data_error = str(e)

def classify_traffic(volume):
    """Classify predicted traffic volume using dataset thresholds."""

    if volume <= 1192.5:
        return "Low"
    elif volume <= 3379:
        return "Moderate"
    elif volume <= 4933:
        return "High"
    else:
        return "Very High"


def show_header(title, description):
    """Display a common page heading."""

    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_data_error():
    if data_error:
        st.error(
            "Could not load the dataset. Check the CSV filename "
            "and make sure it is in the project folder."
        )


if page == "Home":

    show_header(
        "Smart Traffic Analytics",
        "Traffic Analysis & Congestion Prediction"
    )

    st.write(
        "Welcome to the Smart Traffic Analytics system. "
        "This application uses a trained Random Forest model "
        "to estimate traffic volume and explore historical patterns."
    )

    # Banner image
    st.image(
        "https://images.unsplash.com/photo-1519608487953-e999c86e7455"
        "?auto=format&fit=crop&w=1400&q=85",
        caption="Urban transportation and traffic systems",
        use_container_width=True
    )

    st.subheader("Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Traffic Prediction")
        st.write(
            "Enter weather conditions and time-related details "
            "to estimate traffic volume using machine learning."
        )

        st.markdown("### Historical Analytics")
        st.write(
            "Explore traffic patterns by hour, weekday, "
            "weather condition, and month."
        )

    with col2:
        st.markdown("### Machine Learning")
        st.write(
            "The application uses a trained Random Forest "
            "regression model with a saved preprocessing pipeline."
        )

        st.markdown("### Weather Analysis")
        st.write(
            "Explore how traffic volume varies across "
            "different weather conditions."
        )

    st.divider()

    st.subheader("Dataset Overview")

    if df is not None:
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Records",
            f"{len(df):,}"
        )

        col2.metric(
            "Average Traffic",
            f"{df['traffic_volume'].mean():,.0f}"
        )

        col3.metric(
            "Maximum Traffic",
            f"{df['traffic_volume'].max():,.0f}"
        )
    else:
        show_data_error()

    st.info(
        "Use the sidebar to navigate between the pages "
        "and explore the application."
    )


elif page == "Traffic Prediction":

    show_header(
        "Traffic Volume Prediction",
        "Estimate traffic volume using weather and time inputs."
    )

    if model is None:
        st.error(f"Model loading failed: {model_error}")

    else:
        st.write(
            "Enter the required information below "
            "and click Predict Traffic."
        )

        # Get categories learned by the fitted encoder.
        encoder = preprocessor.named_transformers_["cat"]
        holiday_options = list(encoder.categories_[0])
        weather_options = list(encoder.categories_[1])

        # Replace missing-like categories with a readable option.
        holiday_options = [
            "None" if pd.isna(value) else str(value)
            for value in holiday_options
        ]

        weather_options = [
            str(value) for value in weather_options
            if not pd.isna(value)
        ]

        # Input form
        with st.form("traffic_prediction_form"):

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(" Weather Details")

                temp = st.number_input(
                    "Temperature (Kelvin)",
                    min_value=200.0,
                    max_value=330.0,
                    value=300.15,
                    step=0.5
                )

                rain = st.number_input(
                    "Rain (mm)",
                    min_value=0.0,
                    value=0.0,
                    step=0.1
                )

                snow = st.number_input(
                    "Snow (mm)",
                    min_value=0.0,
                    value=0.0,
                    step=0.1
                )

                clouds = st.slider(
                    "Cloud Coverage (%)",
                    min_value=0,
                    max_value=100,
                    value=40
                )

                weather = st.selectbox(
                    "Weather Condition",
                    options=weather_options
                )

            with col2:
                st.subheader("🕒 Time Details")

                hour = st.slider(
                    "Hour of Day",
                    min_value=0,
                    max_value=23,
                    value=17
                )

                day_of_week = st.selectbox(
                    "Day of Week",
                    options=list(range(7)),
                    format_func=lambda x: [
                        "Monday", "Tuesday", "Wednesday",
                        "Thursday", "Friday", "Saturday",
                        "Sunday"
                    ][x]
                )

                month = st.selectbox(
                    "Month",
                    options=list(range(1, 13)),
                    format_func=lambda x: [
                        "January", "February", "March",
                        "April", "May", "June",
                        "July", "August", "September",
                        "October", "November", "December"
                    ][x - 1]
                )

                is_weekend = st.selectbox(
                    "Weekend?",
                    options=[0, 1],
                    format_func=lambda x: (
                        "Yes" if x == 1 else "No"
                    )
                )

                rush_hour = st.selectbox(
                    "Rush Hour?",
                    options=[0, 1],
                    format_func=lambda x: (
                        "Yes" if x == 1 else "No"
                    )
                )

                holiday = st.selectbox(
                    "Holiday",
                    options=holiday_options
                )

            submitted = st.form_submit_button(
                "🚀 Predict Traffic",
                type="primary",
                use_container_width=True
            )

        if submitted:
            new_traffic = pd.DataFrame([{
                "temp": temp,
                "rain_1h": rain,
                "snow_1h": snow,
                "clouds_all": clouds,
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": is_weekend,
                "rush_hour": rush_hour,
                "holiday": holiday,
                "weather_main": weather
            }])

            try:
                processed_data = preprocessor.transform(
                    new_traffic
                )

                prediction = model.predict(
                    processed_data
                )[0]

                traffic_level = classify_traffic(
                    prediction
                )

                st.divider()
                st.subheader(" Prediction Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Estimated Traffic Volume",
                        f"{prediction:,.0f} vehicles"
                    )

                with col2:
                    st.metric(
                        "Traffic Volume Category",
                        traffic_level
                    )

                if traffic_level == "Low":
                    st.success(
                        "The predicted traffic volume is low."
                    )
                elif traffic_level == "Moderate":
                    st.info(
                        "The predicted traffic volume is moderate."
                    )
                elif traffic_level == "High":
                    st.warning(
                        "The predicted traffic volume is high."
                    )
                else:
                    st.error(
                        "The predicted traffic volume is very high."
                    )

                st.caption(
                    "These categories are based on historical "
                    "traffic-volume thresholds, not verified "
                    "road-capacity or congestion measurements."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")


elif page == "Analytics":

    show_header(
        "Traffic Analytics Dashboard",
        "Explore historical traffic patterns and trends."
    )

    if df is None:
        show_data_error()

    else:
        # Summary metrics
        st.subheader("Traffic Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Records",
            f"{len(df):,}"
        )

        col2.metric(
            "Average Traffic",
            f"{df['traffic_volume'].mean():,.0f}"
        )

        col3.metric(
            "Maximum Traffic",
            f"{df['traffic_volume'].max():,.0f}"
        )

        st.divider()

        # Two-column chart layout
        col1, col2 = st.columns(2)

        # Hourly traffic
        with col1:
            st.subheader("Traffic by Hour")

            hourly_traffic = (
                df.groupby("hour")["traffic_volume"]
                .mean()
                .reindex(range(24))
            )

            st.line_chart(
                hourly_traffic,
                use_container_width=True
            )

        # Daily traffic
        with col2:
            st.subheader("Traffic by Day")

            day_names = [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday",
                "Sunday"
            ]

            daily_traffic = (
                df.groupby("day_of_week")["traffic_volume"]
                .mean()
                .reindex(range(7))
            )

            daily_traffic.index = day_names

            st.bar_chart(
                daily_traffic,
                use_container_width=True
            )

        st.divider()

        col1, col2 = st.columns(2)

        # Weather traffic
        with col1:
            st.subheader("Traffic by Weather")

            weather_traffic = (
                df.groupby("weather_main")["traffic_volume"]
                .mean()
                .sort_values(ascending=False)
            )

            st.bar_chart(
                weather_traffic,
                use_container_width=True
            )

        # Monthly traffic
        with col2:
            st.subheader("Traffic by Month")

            monthly_traffic = (
                df.groupby("month")["traffic_volume"]
                .mean()
                .reindex(range(1, 13))
            )

            monthly_traffic.index = [
                "Jan", "Feb", "Mar", "Apr",
                "May", "Jun", "Jul", "Aug",
                "Sep", "Oct", "Nov", "Dec"
            ]

            st.line_chart(
                monthly_traffic,
                use_container_width=True
            )

        st.divider()

        # Traffic distribution
        st.subheader("Traffic Volume Distribution")

        st.bar_chart(
            df["traffic_volume"]
            .value_counts(bins=10)
            .sort_index(),
            use_container_width=True
        )

        st.info(
            "These charts describe historical patterns and "
            "relationships. They do not establish that a "
            "particular factor causes traffic changes."
        )



st.markdown(
    """
    <div class="footer">
        Smart Traffic Analytics | Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)