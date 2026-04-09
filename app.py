import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

st.set_page_config(page_title="Social Media & Mental Health Dashboard", layout="wide")

# Custom CSS for clean UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #6C63FF;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-title { font-size: 1.1em; color: #555; }
    .metric-value { font-size: 2em; font-weight: bold; color: #1A1A2E; }
    h1, h2, h3 { color: #1A1A2E; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Social Media & Mental Health Impact Analysis")
st.markdown("Analyze the true impact of screen time on anxiety and depression, comparing Male vs Female profiles.")

@st.cache_data
def load_data():
    df = pd.read_excel("social_media_mental_health.xlsx")
    df['Gender'] = df['Gender'].str.strip().str.title()
    df = df[df['Gender'].isin(['Male', 'Female'])]
    df['High_Risk'] = ((df['GAD_7_Score'] >= 10) | (df['PHQ_9_Score'] >= 10)).astype(int)
    df['High_Risk_Label'] = df['High_Risk'].map({1: 'High Risk', 0: 'Low Risk'})
    return df

df = load_data()

# SIDEBAR FILTERS
st.sidebar.header("Filter Data")
genders = st.sidebar.multiselect("Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
archetypes = st.sidebar.multiselect("User Archetype", options=df['User_Archetype'].unique(), default=df['User_Archetype'].unique())
contents = st.sidebar.multiselect("Content Type", options=df['Dominant_Content_Type'].unique(), default=df['Dominant_Content_Type'].unique())

filtered_df = df[(df['Gender'].isin(genders)) & 
                 (df['User_Archetype'].isin(archetypes)) & 
                 (df['Dominant_Content_Type'].isin(contents))]

# METRIC CARDS
st.markdown("### Key Metrics")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Total Users</div><div class="metric-value">{len(filtered_df):,}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">High Risk %</div><div class="metric-value">{(filtered_df["High_Risk"].mean()*100):.1f}%</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Screen Time (hrs)</div><div class="metric-value">{filtered_df["Daily_Screen_Time_Hours"].mean():.1f}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Sleep (hrs)</div><div class="metric-value">{filtered_df["Sleep_Duration_Hours"].mean():.1f}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# GENDER COMPARISON
st.markdown("### Gender Comparison")
col1, col2 = st.columns(2)
with col1:
    fig = px.pie(filtered_df, names='Gender', title='Gender Distribution', color_discrete_sequence=['#6C63FF', '#FF6584'])
    st.plotly_chart(fig, use_container_width=True)
with col2:
    risk_by_gender = filtered_df.groupby('Gender')['High_Risk'].mean().reset_index()
    fig = px.bar(risk_by_gender, x='Gender', y='High_Risk', title='High Risk % by Gender', color='Gender', color_discrete_sequence=['#6C63FF', '#FF6584'])
    fig.update_layout(yaxis_tickformat='.1%')
    st.plotly_chart(fig, use_container_width=True)

# SCREEN TIME VS MENTAL HEALTH
st.markdown("### Screen Time vs Mental Health")
col1, col2 = st.columns(2)
with col1:
    fig = px.scatter(filtered_df, x='Daily_Screen_Time_Hours', y='GAD_7_Score', color='Gender', trendline="ols", title='Screen Time vs Anxiety (GAD-7)', color_discrete_sequence=['#6C63FF', '#FF6584'], opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.scatter(filtered_df, x='Daily_Screen_Time_Hours', y='PHQ_9_Score', color='Gender', trendline="ols", title='Screen Time vs Depression (PHQ-9)', color_discrete_sequence=['#6C63FF', '#FF6584'], opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

# ARCHETYPES & CONTENT
st.markdown("### Behavior Analysis")
col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(filtered_df, x='User_Archetype', color='High_Risk_Label', barmode='group', title="Risk by User Archetype", color_discrete_sequence=['#4CAF50', '#FF5252'])
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.histogram(filtered_df, y='Dominant_Content_Type', color='High_Risk_Label', barmode='group', title="Risk by Content Type", color_discrete_sequence=['#4CAF50', '#FF5252'])
    st.plotly_chart(fig, use_container_width=True)

# RISK FACTORS
st.markdown("### Specific Risk Factors")
col1, col2 = st.columns(2)
with col1:
    fig = px.box(filtered_df, x='Late_Night_Usage', y='GAD_7_Score', color='Gender', title="Impact of Late Night Usage on Anxiety", color_discrete_sequence=['#6C63FF', '#FF6584'])
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.box(filtered_df, x='Social_Comparison_Trigger', y='PHQ_9_Score', color='Gender', title="Social Comparison vs Depression", color_discrete_sequence=['#6C63FF', '#FF6584'])
    st.plotly_chart(fig, use_container_width=True)

# CORRELATION
st.markdown("### Correlation Matrix")
num_cols = ['Age', 'Daily_Screen_Time_Hours', 'Sleep_Duration_Hours', 'GAD_7_Score', 'PHQ_9_Score']
corr = filtered_df[num_cols].corr()
fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdYlBu_r', title="Feature Correlation Heatmap")
st.plotly_chart(fig, use_container_width=True)

# MODEL PERFORMANCE
st.markdown("### Machine Learning Model Performance")
st.markdown("Test Set results identifying `High_Risk` users:")
model_data = {
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Decision Tree', 'KNN (k=5)', 'Neural Network'],
    'Accuracy': ['81.4%', '80.7%', '79.9%', '70.6%', '71.3%', '85.4%'],
    'Precision': ['79.6%', '79.7%', '78.2%', '69.7%', '72.3%', '82.4%'],
    'Recall (Most Important)': ['82.7%', '80.5%', '81.2%', '69.7%', '66.5%', '81.6%'],
    'F1-Score': ['81.1%', '80.1%', '79.7%', '69.7%', '69.3%', '82.0%']
}
st.table(pd.DataFrame(model_data))

# FEATURE IMPORTANCES
st.markdown("### Gender-Specific Predictive Drivers (Random Forest)")
col1, col2 = st.columns(2)
if os.path.exists("output/task4_gender_feature_importances.png"):
    st.image("output/task4_gender_feature_importances.png", use_container_width=True)
else:
    st.info("Feature importance graphic will appear here once generated by the pipeline.")

# CLUSTERING & ASSOCIATION
st.markdown("### Advanced Analytics: Clustering & Association Rules")
col1, col2 = st.columns(2)
with col1:
    if os.path.exists("output/task6_pca_2d.png"):
        st.image("output/task6_pca_2d.png", caption="K-Means Clusters", use_container_width=True)
with col2:
    if os.path.exists("output/task7_association_rules.png"):
        st.image("output/task7_association_rules.png", caption="Behavioral Association Rules (Lift > 1.5)", use_container_width=True)

# RECOMMENDATIONS
st.markdown("---")
st.markdown("### 📋 Executive Recommendations")
st.markdown("""
**1. Association Rules**  
- **Problem:** High screen time combined with Late night browsing correlates with severe sleep deprivation.  
- **Solution:** Introduce automated dark-mode lockouts or aggressive wind-down notifications post-midnight.  
- **Expected Impact:** Decreases sleep onset latency, reducing compound anxiety scores naturally.

**2. Clustering**  
- **Problem:** Significant hidden baseline of "High-Risk Addicts" currently masked as standard heavy users.  
- **Solution:** Proactively route identified Cluster-2 users directly to active psychiatric support/therapy funnels via the UI.  
- **Expected Impact:** Enhances mental health triage efficiency.

**3. Gender Bias**  
- **Problem:** Female users incur far more harm from comparative, lifestyle-centric visual feeds.  
- **Solution:** Implement algorithms that forcefully inject neutral, non-competitive content (art, comedy) after 45 mins.  
- **Expected Impact:** Interrupts the 'Social Comparison Trigger' loop minimizing trait-anxiety accumulation.

**4. Feature Importance**  
- **Problem:** Social Comparison represents the strongest Random Forest predictor strictly for women.  
- **Solution:** Opt-in / Default-on hiding of public like-counts and follower metrics for identified at-risk users.  
- **Expected Impact:** Statistically removes the largest antecedent risk factor triggering PHQ-9.

**5. High Risk Patterns**  
- **Problem:** Clinical severity directly scales concurrently with Daily Screen Volume regardless of content for Males.  
- **Solution:** Hard UI circuit-breakers (e.g., 20-min necessary offline break) gating access after 6 cumulative hours.  
- **Expected Impact:** Restores usage volume back to 'Healthy User' parameters.
""")
