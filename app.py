import gradio as gr
import pandas as pd
import pickle
import plotly.express as px
import shap
import matplotlib.pyplot as plt

from utils.preprocessing import load_data
from utils.resource_allocator import allocate_resources
from utils.chatbot import health_chatbot

# Load datasets
df = load_data()

hospital_df = pd.read_excel(
    "dataset/tamilnadu_hospital_resources_dengue_project.xlsx"
)

# Load India dengue data
india_df_raw = pd.read_csv("dataset/dengue_cases_in_india.csv")

# Transform to long format
years = [2019, 2020, 2021, 2022, 2023, "2024*"]
cases_cols = [f"{year}_Cases" for year in years]
india_df = india_df_raw.melt(id_vars=["States"], value_vars=cases_cols, var_name="year", value_name="cases")
india_df["year"] = india_df["year"].str.replace("_Cases", "").str.replace("*", "").astype(int)
india_df.rename(columns={"States": "state"}, inplace=True)

# Clean state names and cases column
india_df["state"] = india_df["state"].str.replace("*", "", regex=False).str.strip()
india_df["cases"] = india_df["cases"].replace("NR", 0).astype(int)

# Load trained models
models = pickle.load(open("model/models.pkl", "rb"))


# -----------------------------
# Visualization Functions 
# -----------------------------

def dengue_trend():

    fig = px.line(
        df,
        x="weekofyear",
        y="total_cases",
        color="city",
        title="Dengue Cases Over Time"
    )

    fig.update_layout(template="plotly_dark")

    return fig


def temperature_vs_cases():

    fig = px.scatter(
        df,
        x="station_avg_temp_c",
        y="total_cases",
        color="city",
        title="Temperature vs Dengue Cases"
    )

    fig.update_layout(template="plotly_dark")

    return fig


def hospital_resource_chart():

    fig = px.bar(
        hospital_df,
        x="hospital_name",
        y="available_beds",
        color="city",
        title="Hospital Bed Availability"
    )

    fig.update_layout(template="plotly_dark")

    return fig


# -----------------------------
# India Dashboard Functions
# -----------------------------

def india_kpis():
    total_cases = int(india_df["cases"].sum())
    avg_cases = int(india_df["cases"].mean())
    max_state = india_df.groupby("state")["cases"].sum().idxmax()

    return f"""
### 📊 India Dengue Intelligence Dashboard

<div style="display:flex; flex-wrap:wrap; gap:18px; margin-bottom:20px;">
    <div style="flex:1; min-width:220px; background:#0f172a; padding:20px; border-radius:18px; box-shadow:0 18px 36px rgba(15,23,42,0.35);">
        <h4 style="margin:0;color:#fb7185;">🔴 Total Cases</h4>
        <h2 style="margin:10px 0 0;color:#f8fafc;">{total_cases:,}</h2>
    </div>
    <div style="flex:1; min-width:220px; background:#0f172a; padding:20px; border-radius:18px; box-shadow:0 18px 36px rgba(15,23,42,0.35);">
        <h4 style="margin:0;color:#facc15;">🟡 Avg Cases</h4>
        <h2 style="margin:10px 0 0;color:#f8fafc;">{avg_cases:,}</h2>
    </div>
    <div style="flex:1; min-width:220px; background:#0f172a; padding:20px; border-radius:18px; box-shadow:0 18px 36px rgba(15,23,42,0.35);">
        <h4 style="margin:0;color:#34d399;">🏆 Highest State</h4>
        <h2 style="margin:10px 0 0;color:#f8fafc;">{max_state}</h2>
    </div>
</div>
"""

def india_map(year_filter=None, max_cases=None):
    filtered = india_df.copy()
    if year_filter is not None:
        filtered = filtered[filtered["year"] == year_filter]
    if max_cases is not None:
        filtered = filtered[filtered["cases"] <= max_cases]

    state_coords = {
        "A& N Island": (11.7, 92.7),
        "Andhra Pradesh": (15.9, 79.7),
        "Arunachal Pradesh": (28.2, 94.6),
        "Assam": (26.7, 92.8),
        "Bihar": (25.1, 85.3),
        "Chandigarh": (30.7, 76.8),
        "Chattisgarh": (21.8, 82.3),
        "D&N Haveli": (20.3, 73.0),
        "Daman & Diu": (20.4, 72.8),
        "Delhi": (28.6, 77.2),
        "Goa": (15.4, 73.8),
        "Gujarat": (22.3, 71.8),
        "Haryana": (29.1, 76.0),
        "Himachal Pradesh": (31.1, 77.2),
        "J & K": (33.8, 75.1),
        "Jharkhand": (23.6, 85.3),
        "Karnataka": (15.3, 76.6),
        "Kerala": (10.8, 76.9),
        "Lakshadweep": (10.5, 72.6),
        "Madhya Pradesh": (22.7, 78.0),
        "Maharashtra": (19.7, 75.3),
        "Manipur": (24.7, 93.9),
        "Meghalaya": (25.6, 91.9),
        "Mizoram": (23.3, 92.7),
        "Nagaland": (26.0, 94.5),
        "Odisha": (20.9, 84.9),
        "Puduchery": (11.9, 79.8),
        "Punjab": (31.2, 75.5),
        "Rajasthan": (27.0, 74.2),
        "Sikkim": (27.5, 88.5),
        "Tamil Nadu": (11.0, 78.7),
        "Telangana": (18.0, 79.5),
        "Tripura": (23.8, 91.3),
        "Uttar Pradesh": (26.8, 80.9),
        "Uttrakhand": (30.0, 79.0),
        "West Bengal": (22.5, 88.3)
    }

    filtered = filtered.copy()
    filtered["latitude"] = filtered["state"].map(lambda s: state_coords.get(s, (20.6, 78.9))[0])
    filtered["longitude"] = filtered["state"].map(lambda s: state_coords.get(s, (20.6, 78.9))[1])

    fig = px.scatter_geo(
        filtered,
        lat="latitude",
        lon="longitude",
        color="cases",
        size="cases",
        hover_name="state",
        hover_data={"cases": True, "year": True, "latitude": False, "longitude": False},
        projection="natural earth",
        title="🗺️ Dengue Cases Across India",
        color_continuous_scale="Reds",
        size_max=30
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        geo_scope="asia",
        template="plotly_dark",
        margin=dict(l=0, r=0, t=45, b=0),
        legend_title_text="Cases"
    )
    return fig


def india_trend_animation():
    fig = px.line(
        india_df,
        x="year",
        y="cases",
        color="state",
        line_group="state",
        animation_frame="year",
        markers=True,
        title="📈 Dengue Spread Over Years"
    )
    fig.update_layout(template="plotly_dark", showlegend=False, margin=dict(l=0, r=0, t=45, b=0))
    fig.update_traces(marker=dict(size=8))
    return fig


def state_resource_planner(state):
    state_data = india_df[india_df["state"] == state]
    total_cases = int(state_data["cases"].sum())

    if total_cases > 50000:
        risk = "🔴 HIGH"
    elif total_cases > 20000:
        risk = "🟡 MEDIUM"
    else:
        risk = "🟢 LOW"

    predicted_cases = int(total_cases * 1.12 + 500)
    beds = int(predicted_cases * 0.2)
    doctors = int(predicted_cases * 0.05)
    icu = int(predicted_cases * 0.03)
    staff = int(predicted_cases * 0.1)

    insight = f"📌 Insight: Dengue cases in {state} show an increasing trend in recent years, likely driven by humidity and rainfall patterns."

    return f"""
### 📍 State: {state}
**Risk Level:** {risk}

### 🦠 Total Dengue Cases (Historic): {total_cases}

### 📈 Prediction for Next Year
**Estimated Cases:** {predicted_cases}
**Beds Needed:** {beds}
**Doctors Required:** {doctors}
**ICU Beds Needed:** {icu}
**Medical Staff Required:** {staff}

### 🧠 Smart Insight
{insight}
"""

def state_trend(state):
    df_state = india_df[india_df["state"] == state]

    fig = px.line(
        df_state,
        x="year",
        y="cases",
        title=f"{state} Dengue Trend"
    )

    return fig


def top_states(year=None, max_cases=None):
    filtered = india_df.copy()
    if year is not None:
        filtered = filtered[filtered["year"] == year]
    if max_cases is not None:
        filtered = filtered[filtered["cases"] <= max_cases]

    top5 = filtered.groupby("state")["cases"].sum().nlargest(5).reset_index()
    top5["rank"] = range(1, len(top5) + 1)

    fig = px.bar(
        top5,
        x="cases",
        y="state",
        orientation="h",
        color="cases",
        color_continuous_scale="plasma",
        title="🔥 Top 5 High Risk States"
    )
    fig.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=45, b=0))
    fig.update_traces(text=top5["rank"], textposition="inside", hovertemplate="%{y}: %{x} cases<extra></extra>")
    return fig


def global_tb_map():
    tb_data = pd.DataFrame({
        "country": [
            "India", "Indonesia", "Brazil", "Nigeria", "South Africa",
            "Pakistan", "Bangladesh", "Ethiopia", "Philippines", "China"
        ],
        "TB_Rate": [21000, 19000, 13000, 23000, 25000, 16000, 14000, 18000, 17000, 12000]
    })

    fig = px.choropleth(
        tb_data,
        locations="country",
        locationmode="country names",
        color="TB_Rate",
        hover_name="country",
        color_continuous_scale="Reds",
        title="Global TB Incidence Rate"
    )
    fig.update_traces(marker_line_width=0.2, marker_line_color="white")
    fig.update_geos(showframe=False, showcoastlines=False)
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=45, b=0),
        coloraxis_colorbar=dict(title="TB Rate")
    )
    return fig


# -----------------------------
# Prediction Function
# -----------------------------

def predict_cases(model_name, temp_k, humidity, rainfall, station_temp):

    model = models[model_name]

    sample = [[temp_k, humidity, rainfall, station_temp]]

    prediction = float(model.predict(sample)[0])

    status, beds, doctors = allocate_resources(prediction)

    result = f"""
### 🧾 Prediction Result

**Predicted Dengue Cases:** {prediction:.2f}

**Resource Status:** {status}

**Beds Needed:** {beds}

**Doctors Needed:** {doctors}
"""

    return result


# ------------------------------
# Patient Symptom Analysis Function
# ------------------------------

def analyze_patient_symptoms(age, fever, headache, nausea, joint_pain, rash, body_ache, additional_notes):
    """
    Analyze patient symptoms and provide dengue risk assessment
    """
    
    # Count symptoms
    symptom_count = sum([fever, headache, nausea, joint_pain, rash, body_ache])
    
    # Age risk factor
    if age < 5 or age > 65:
        age_risk = "🔴 HIGH (Children and elderly at higher risk)"
    elif age < 18 or age > 50:
        age_risk = "🟡 MEDIUM"
    else:
        age_risk = "🟢 LOW"
    
    # Symptom risk assessment
    if symptom_count == 0:
        symptom_risk = "🟢 LOW (No dengue symptoms detected)"
        condition = "Asymptomatic or No Symptoms"
    elif symptom_count <= 2:
        symptom_risk = "🟡 MILD (Few symptoms - Monitor closely)"
        condition = "Mild Symptoms - Observation Required"
    elif symptom_count <= 4:
        symptom_risk = "🟠 MODERATE (Multiple symptoms present)"
        condition = "Moderate Severity - Medical Attention Recommended"
    else:
        symptom_risk = "🔴 SEVERE (Many symptoms present)"
        condition = "Severe - Immediate Medical Attention Required"
    
    # Detailed symptom report
    symptoms_detected = []
    if fever:
        symptoms_detected.append("• Fever (High temperature)")
    if headache:
        symptoms_detected.append("• Headache (Severe head pain)")
    if nausea:
        symptoms_detected.append("• Nausea (Feeling of sickness)")
    if joint_pain:
        symptoms_detected.append("• Joint Pain (Muscle & bone ache)")
    if rash:
        symptoms_detected.append("• Rash (Skin manifestation)")
    if body_ache:
        symptoms_detected.append("• Body Ache (General discomfort)")
    
    result = f"""
### 🧾 Patient Symptom Analysis Report

**Patient Age:** {int(age)} years old  
**Age Risk Factor:** {age_risk}

---

### 🦠 Symptoms Summary

**Symptoms Detected:** {symptom_count}/6

{chr(10).join(symptoms_detected) if symptoms_detected else "• No symptoms recorded"}

---

### ⚠️ Dengue Risk Assessment

**Symptom Risk Level:** {symptom_risk}

**Clinical Condition:** {condition}

---

### 📋 Recommendation

"""
    
    if symptom_count == 0:
        result += """
**Action:** Continue monitoring for any symptoms. Maintain hygiene and mosquito prevention measures.

**Follow-up:** If symptoms develop, seek medical attention immediately.
"""
    elif symptom_count <= 2:
        result += """
**Action:** Monitor symptoms closely. Rest well and stay hydrated.

**Follow-up:** Consult a healthcare professional if symptoms worsen or new symptoms appear.

**Lab Test:** Consider dengue diagnostic testing if symptoms persist beyond 2 days.
"""
    elif symptom_count <= 4:
        result += """
**Action:** **SEEK MEDICAL ATTENTION SOON**. Consult a healthcare provider immediately.

**Lab Test:** Urgent dengue diagnostic testing (NS1, IgM, PCR) recommended.

**Hospitalization:** May require observation and supportive care.

**Resources:** Bed allocation recommended based on severity.
"""
    else:
        result += """
**Action:** **EMERGENCY - IMMEDIATE MEDICAL ATTENTION REQUIRED**

**Lab Test:** PRIORITY dengue diagnostic testing & blood work needed.

**Hospitalization:** High probability of hospitalization required.

**Resources Allocated:** ICU bed may be necessary.

**Alert:** Contact emergency services or nearest hospital immediately.
"""
    
    if additional_notes:
        result += f"\n\n**Additional Patient Notes:** {additional_notes}"
    
    return result


# -----------------------------
# Explainable AI
# -----------------------------

def explain_model():

    model = models["Random Forest"]

    X = df[[
        "reanalysis_air_temp_k",
        "reanalysis_specific_humidity_g_per_kg",
        "precipitation_amt_mm",
        "station_avg_temp_c"
    ]]

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    plt.figure()

    shap.summary_plot(shap_values, X, show=False)

    return plt.gcf()


# ---------------------------------------
# Prediction + Resource Planning Panel
# ---------------------------------------

def dengue_resource_planner(year, temperature, humidity, rainfall):

    model = models["Random Forest"]

    sample = [[temperature, humidity, rainfall, temperature]]

    predicted_cases = int(model.predict(sample)[0])

    beds = int(predicted_cases * 0.2)
    doctors = int(predicted_cases * 0.05)
    icu = int(predicted_cases * 0.03)
    staff = int(predicted_cases * 0.1)

    result = f"""
### Prediction Result

**Predicted Dengue Cases:** {predicted_cases}

### Recommended Healthcare Resources

Hospital Beds Needed: {beds}

Doctors Required: {doctors}

ICU Beds Needed: {icu}

Medical Staff Required: {staff}
"""

    return result


# -----------------------------
# Dashboard UI
# -----------------------------

with gr.Blocks() as demo:

    gr.Markdown(
        """
        # 🏥 AI Dengue Monitoring & Resource Allocation System
        ### Healthcare Analytics Dashboard
        """
    )

    with gr.Tabs():

        # --------------------------------------------------
        # DATASET
        # --------------------------------------------------

        with gr.Tab("📊 Dataset Overview"):

            gr.Markdown("### Dengue Dataset Sample")

            gr.Dataframe(
                df.head(50)
            )

        # --------------------------------------------------
        # PATIENT INPUT
        # --------------------------------------------------

        with gr.Tab("🧑 Patient Symptoms"):

            gr.Markdown("## 🏥 Patient Symptom Assessment & Risk Evaluation")
            
            gr.Markdown(
                """
                ### Welcome to the Dengue Symptom Checker
                Please provide your medical information below for a comprehensive dengue risk assessment.
                Our AI system will analyze your symptoms and provide personalized recommendations.
                """
            )

            gr.Markdown("### 👤 Patient Information")
            with gr.Group():
                with gr.Row():
                    age = gr.Number(
                        label="Age", 
                        value=25, 
                        info="Patient's current age in years",
                        minimum=0,
                        maximum=120
                    )

            gr.Markdown("### 🦠 Dengue Symptoms Checklist")
            gr.Markdown("**Select all symptoms you are experiencing:**")
            with gr.Group():
                with gr.Row():
                    fever = gr.Checkbox(
                        label="🌡️ Fever",
                        info="High body temperature (>38°C)"
                    )
                    headache = gr.Checkbox(
                        label="🤕 Headache",
                        info="Severe head pain"
                    )
                    nausea = gr.Checkbox(
                        label="🤢 Nausea",
                        info="Feeling of sickness/vomiting"
                    )

                with gr.Row():
                    joint_pain = gr.Checkbox(
                        label="🦴 Joint Pain",
                        info="Muscle and bone aches"
                    )
                    rash = gr.Checkbox(
                        label="🔴 Rash",
                        info="Skin rash or discoloration"
                    )
                    body_ache = gr.Checkbox(
                        label="💪 Body Ache",
                        info="General body discomfort"
                    )

            gr.Markdown("### 📝 Additional Information")
            with gr.Group():
                additional_notes = gr.Textbox(
                    label="Additional Medical Notes",
                    placeholder="Enter any additional symptoms or medical history...",
                    lines=3,
                    info="Optional: Provide any other relevant medical information"
                )

            gr.Markdown("---")

            with gr.Row():
                submit_btn = gr.Button(
                    "🔍 Analyze Symptoms & Get Assessment",
                    variant="primary",
                    scale=2
                )
                clear_btn = gr.Button(
                    "🔄 Clear Form",
                    scale=1
                )

            gr.Markdown("### 📊 Analysis Result")
            
            analysis_output = gr.Markdown(
                value="""
                ### 📊 Waiting for Input
                Click **"Analyze Symptoms & Get Assessment"** to see your personalized dengue risk assessment.
                """
            )

            # Handler for analyze button
            submit_btn.click(
                analyze_patient_symptoms,
                inputs=[age, fever, headache, nausea, joint_pain, rash, body_ache, additional_notes],
                outputs=analysis_output
            )

            # Handler for clear button
            def reset_form():
                return 25, False, False, False, False, False, False, ""

            clear_btn.click(
                reset_form,
                outputs=[age, fever, headache, nausea, joint_pain, rash, body_ache, additional_notes]
            )

            gr.Markdown(
                """
                ---
                ### ⚠️ Disclaimer
                This tool is for educational and informational purposes only. 
                It is **NOT** a substitute for professional medical advice.
                Always consult with a qualified healthcare provider for accurate diagnosis and treatment.
                """
            )

        # --------------------------------------------------
        # VISUALIZATION
        # --------------------------------------------------

        with gr.Tab("📈 Disease Spread Visualization"):

            gr.Plot(dengue_trend)

            gr.Plot(temperature_vs_cases)

        # --------------------------------------------------
        # GLOBAL MAP FEATURE
        # --------------------------------------------------

        with gr.Tab("🌍 Global Disease Map"):

            gr.Markdown("### Global TB Incidence Rate")
            gr.Plot(global_tb_map)

        # --------------------------------------------------
        # ML PREDICTION
        # --------------------------------------------------

        with gr.Tab("🤖 Dengue Prediction Model"):

            gr.Markdown("### Enter Environmental Data")

            with gr.Row():

                model_select = gr.Dropdown(
                    ["Random Forest", "Linear Regression", "XGBoost"],
                    label="Select ML Model"
                )

            with gr.Row():

                temp_k = gr.Number(label="Air Temperature (K)")

                humidity = gr.Number(label="Humidity")

                rainfall = gr.Number(label="Rainfall")

                station_temp = gr.Number(label="Station Temperature (C)")

            predict_button = gr.Button("🚀 Predict Dengue Cases")

            prediction_output = gr.Markdown()

            predict_button.click(
                predict_cases,
                inputs=[
                    model_select,
                    temp_k,
                    humidity,
                    rainfall,
                    station_temp
                ],
                outputs=prediction_output
            )

        # --------------------------------------------------
        # Prediction & Resource Planning
        # --------------------------------------------------

        with gr.Tab("Prediction & Resource Planning"):

            gr.Markdown("## Predict Future Dengue Cases")

            year_input = gr.Number(
                label="Enter Year",
                value=2026
            )

            temp_input = gr.Number(
                label="Average Temperature (C)"
            )

            humidity_input = gr.Number(
                label="Humidity Level"
            )

            rainfall_input = gr.Number(
                label="Rainfall Level"
            )

            planner_output = gr.Markdown()

            predict_btn = gr.Button(
                "Predict Dengue Cases & Resources"
            )

            predict_btn.click(
                dengue_resource_planner,
                inputs=[
                    year_input,
                    temp_input,
                    humidity_input,
                    rainfall_input
                ],
                outputs=planner_output
            )

        # --------------------------------------------------
        # RESOURCE ALLOCATION
        # --------------------------------------------------

        with gr.Tab("🏥 Resource Allocation"):

            gr.Markdown("### Tamil Nadu Hospital Resources")

            gr.Dataframe(
                hospital_df
            )

            gr.Plot(hospital_resource_chart)

        # --------------------------------------------------
        # EXPLAINABLE AI
        # --------------------------------------------------

        with gr.Tab("🧠 Explainable AI"):

            gr.Markdown(
                "### Feature Importance using SHAP"
            )

            explain_button = gr.Button(
                "Generate SHAP Explanation"
            )

            shap_plot = gr.Plot()

            explain_button.click(
                explain_model,
                outputs=shap_plot
            )

        # --------------------------------------------------
        # HEALTH CHATBOT
        # --------------------------------------------------

        with gr.Tab("💬 Dengue Health Assistant"):

            gr.Markdown(
                "Ask questions about Dengue symptoms, prevention and treatment."
            )

            gr.ChatInterface(
                fn=health_chatbot,
                title="Dengue AI Health Assistant"
            )

        # --------------------------------------------------
        # 🇮🇳 India Dengue Intelligence Dashboard
        # --------------------------------------------------

        with gr.Tab("🇮🇳 India Intelligence Dashboard"):

            gr.Markdown("# 🇮🇳 India Dengue Intelligence Dashboard")
            gr.Markdown("---")

            gr.Markdown(india_kpis())

            with gr.Row():
                year_filter = gr.Dropdown(
                    choices=sorted(india_df["year"].unique().tolist()),
                    value=2023,
                    label="Select Year",
                    info="Filter the map and top states by year"
                )
                max_cases_filter = gr.Slider(
                    minimum=0,
                    maximum=int(india_df["cases"].max()),
                    value=int(india_df["cases"].max()),
                    step=500,
                    label="Max Cases",
                    info="Only include states with case counts below this threshold"
                )
                apply_filters = gr.Button("🔎 Apply Filters")

            with gr.Row():
                map_plot = gr.Plot(value=india_map(year_filter.value, max_cases_filter.value))

            with gr.Row():
                top_states_plot = gr.Plot(value=top_states(year_filter.value, max_cases_filter.value))

            with gr.Row():
                animation_plot = gr.Plot(value=india_trend_animation())

            apply_filters.click(
                fn=lambda year, max_cases: (india_map(year, max_cases), top_states(year, max_cases)),
                inputs=[year_filter, max_cases_filter],
                outputs=[map_plot, top_states_plot]
            )

            gr.Markdown("---")
            gr.Markdown("## 📍 State Deep Analysis")

            with gr.Row():
                state_select = gr.Dropdown(
                    choices=india_df["state"].unique().tolist(),
                    label="Select State",
                    value=india_df["state"].unique().tolist()[0]
                )
                analyze_state_btn = gr.Button("Analyze State")

            with gr.Row():
                with gr.Column(scale=2):
                    trend_plot = gr.Plot()
                with gr.Column(scale=1):
                    analysis_output = gr.Markdown(
                        value="### Select a state and click Analyze State to see detailed insights, risk level, and prediction."
                    )

            analyze_state_btn.click(
                fn=lambda state: (
                    state_trend(state),
                    state_resource_planner(state)
                ),
                inputs=state_select,
                outputs=[trend_plot, analysis_output]
            )

import os
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    theme=gr.themes.Soft()
)