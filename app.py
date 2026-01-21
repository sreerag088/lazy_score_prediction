import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Activity Level Assessment",
    page_icon="🎯",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #4338CA;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        # Try to load the CSV file
        df = pd.read_csv('lazyy_data.csv')
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        return df, None
    except FileNotFoundError:
        return None, "File 'lazy_data.csv' not found. Please ensure it's in the same directory as this script."
    except Exception as e:
        return None, f"Error loading CSV: {str(e)}"

# Train model
@st.cache_resource
def train_model(df):
    try:
        # Find column names (handle variations)
        cols = df.columns.tolist()
        
        # Try to identify columns by keywords
        delay_col = [c for c in cols if 'delay' in c.lower()][0]
        unfinished_col = [c for c in cols if 'unfinished' in c.lower() or 'leave' in c.lower()][0]
        routine_col = [c for c in cols if 'routine' in c.lower() or 'fail' in c.lower()][0]
        activity_col = [c for c in cols if 'physical' in c.lower() or 'activity' in c.lower()][0]
        phone_col = [c for c in cols if 'phone' in c.lower() or 'scroll' in c.lower()][0]
        
        # Get unique values from each column to understand the data
        unique_delay = df[delay_col].unique()
        unique_unfinished = df[unfinished_col].unique()
        unique_routine = df[routine_col].unique()
        unique_activity = df[activity_col].unique()
        unique_phone = df[phone_col].unique()
        
        # Create label encoders
        le_delay = LabelEncoder()
        le_unfinished = LabelEncoder()
        le_routine = LabelEncoder()
        le_activity = LabelEncoder()
        le_phone = LabelEncoder()
        
        # Prepare features
        X = pd.DataFrame({
            'delay_tasks': le_delay.fit_transform(df[delay_col].astype(str)),
            'unfinished_tasks': le_unfinished.fit_transform(df[unfinished_col].astype(str)),
            'fail_routine': le_routine.fit_transform(df[routine_col].astype(str)),
            'physical_activity': le_activity.fit_transform(df[activity_col].astype(str)),
            'phone_scrolling': le_phone.fit_transform(df[phone_col].astype(str))
        })
        
        # Calculate laziness score
        scores = []
        for idx in range(len(df)):
            score = 0
            
            # Delay tasks
            val = str(df[delay_col].iloc[idx]).strip()
            if val == 'Always':
                score += 2
            elif val == 'Often':
                score += 1.5
            elif val == 'Sometimes':
                score += 1
            elif val == 'Rarely':
                score += 0.5
                
            # Unfinished tasks
            val = str(df[unfinished_col].iloc[idx]).strip()
            if val == 'Always':
                score += 2
            elif val == 'Often':
                score += 1.5
            elif val == 'Sometimes':
                score += 1
            elif val == 'Rarely':
                score += 0.5
                
            # Fail routine
            val = str(df[routine_col].iloc[idx]).strip()
            if val == 'Always':
                score += 2
            elif val == 'Often':
                score += 1.5
            elif val == 'Sometimes':
                score += 1
            elif val == 'Rarely':
                score += 0.5
                
            # Physical activity (inverse)
            val = str(df[activity_col].iloc[idx]).strip()
            if val == 'Not active':
                score += 2
            elif val == 'Slightly active':
                score += 1.5
            elif val == 'Moderately active':
                score += 1
            elif val == 'Very active':
                score += 0.5
                
            # Phone scrolling
            val = str(df[phone_col].iloc[idx]).strip()
            if val == 'Always':
                score += 2
            elif val == 'Often':
                score += 1.5
            elif val == 'Sometimes':
                score += 1
            elif val == 'Rarely':
                score += 0.5
            
            # Normalize to 1-5 scale
            normalized_score = int(np.clip(np.round(score / 2), 1, 5))
            scores.append(normalized_score)
        
        y = pd.Series(scores)
        
        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)
        
        # Get unique classes
        classes = sorted(y.unique())
        
        return (model, le_delay, le_unfinished, le_routine, le_activity, le_phone, classes, 
                unique_delay, unique_unfinished, unique_routine, unique_activity, unique_phone, None)
    except Exception as e:
        return None, None, None, None, None, None, None, None, None, None, None, None, f"Error training model: {str(e)}"

# Load data
df, load_error = load_data()

# App header
st.title("🎯 Activity Level Assessment")
st.markdown("### Complete this survey to get your activity score prediction")
st.markdown("---")

if load_error:
    st.error(f"⚠️ **{load_error}**")
    st.info("""
    **Instructions:**
    1. Make sure 'lazy_data.csv' is in the same folder as this script
    2. The CSV should have columns for behavioral questions and activity level
    3. Restart the app after placing the file
    """)
    st.stop()

# Train model
result = train_model(df)
model, le_delay, le_unfinished, le_routine, le_activity, le_phone, classes = result[:7]
unique_delay, unique_unfinished, unique_routine, unique_activity, unique_phone, train_error = result[7:]

if train_error:
    st.error(f"⚠️ **{train_error}**")
    st.info("Please check your CSV file structure and data format.")
    st.stop()

# Create the survey form
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 📋 Behavioral Patterns")
    
    delay_tasks = st.selectbox(
        "How often do you delay tasks even when they are simple?",
        options=sorted(unique_delay.tolist()),
        index=None,
        placeholder="Select an option"
    )
    
    unfinished_tasks = st.selectbox(
        "How often do you leave tasks unfinished?",
        options=sorted(unique_unfinished.tolist()),
        index=None,
        placeholder="Select an option"
    )
    
    fail_routine = st.selectbox(
        "How often do you fail to follow a daily routine?",
        options=sorted(unique_routine.tolist()),
        index=None,
        placeholder="Select an option"
    )

with col2:
    st.markdown("#### 🏃 Lifestyle Habits")
    
    physical_activity = st.selectbox(
        "What is your physical activity level?",
        options=sorted(unique_activity.tolist()),
        index=None,
        placeholder="Select an option"
    )
    
    phone_scrolling = st.selectbox(
        "How often do you scroll on your phone instead of working?",
        options=sorted(unique_phone.tolist()),
        index=None,
        placeholder="Select an option"
    )

st.markdown("---")

# Prediction button
if st.button("🔮 Get Your Activity Score"):
    # Check if all fields are filled
    if None in [delay_tasks, unfinished_tasks, fail_routine, physical_activity, phone_scrolling]:
        st.error("⚠️ Please answer all questions before getting your prediction!")
    else:
        try:
            # Prepare input data
            input_data = pd.DataFrame({
                'delay_tasks': [le_delay.transform([delay_tasks])[0]],
                'unfinished_tasks': [le_unfinished.transform([unfinished_tasks])[0]],
                'fail_routine': [le_routine.transform([fail_routine])[0]],
                'physical_activity': [le_activity.transform([physical_activity])[0]],
                'phone_scrolling': [le_phone.transform([phone_scrolling])[0]]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Get prediction probability
            proba = model.predict_proba(input_data)[0]
            confidence = max(proba) * 100
            
            # Define score characteristics
            score_info = {
                1: {"label": "Very Active & Productive", "color": "#10b981", "emoji": "🌟", "bg": "#d1fae5", 
                    "desc": "Excellent! You show strong productivity habits and maintain an active lifestyle."},
                2: {"label": "Active & Engaged", "color": "#84cc16", "emoji": "✨", "bg": "#ecfccb",
                    "desc": "Great! You're generally productive with good activity levels."},
                3: {"label": "Moderate Activity", "color": "#eab308", "emoji": "⚡", "bg": "#fef9c3",
                    "desc": "You have a balanced approach but there's room for improvement."},
                4: {"label": "Low Activity Level", "color": "#f97316", "emoji": "⚠️", "bg": "#fed7aa",
                    "desc": "Consider working on building better habits and increasing activity."},
                5: {"label": "Very Low Activity", "color": "#ef4444", "emoji": "🔴", "bg": "#fecaca",
                    "desc": "Focus on small steps to improve your daily routines and activity level."}
            }
            
            info = score_info.get(prediction, score_info[3])
            
            # Display result
            st.markdown("---")
            st.markdown("### 🎊 Your Results")
            
            st.markdown(f"""
            <div style="background-color: {info['bg']}; padding: 2rem; border-radius: 1rem; 
                        border: 3px solid {info['color']}; text-align: center; margin: 1rem 0;">
                <h1 style="color: {info['color']}; font-size: 4rem; margin: 0;">{info['emoji']}</h1>
                <h2 style="color: {info['color']}; margin: 0.5rem 0;">Activity Score: {prediction}/5</h2>
                <h3 style="color: {info['color']}; margin: 0.5rem 0;">{info['label']}</h3>
                <p style="color: #374151; font-size: 1rem; margin-top: 1rem;">
                    {info['desc']}
                </p>
                <p style="color: #6b7280; font-size: 0.9rem; margin-top: 0.5rem;">
                    Prediction Confidence: {confidence:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("#### 📊 Score Distribution")
            
            # Create columns for all possible classes
            cols = st.columns(len(classes))
            
            for i, col in enumerate(cols):
                with col:
                    class_val = classes[i]
                    class_idx = list(model.classes_).index(class_val)
                    prob_val = proba[class_idx]
                    
                    if class_val == prediction:
                        st.markdown(f"**Score {class_val}**")
                        st.progress(1.0)
                        st.markdown(f"<p style='text-align: center; font-size: 0.8rem;'>{prob_val*100:.1f}%</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"Score {class_val}")
                        st.progress(float(prob_val))
                        st.markdown(f"<p style='text-align: center; font-size: 0.8rem;'>{prob_val*100:.1f}%</p>", unsafe_allow_html=True)
            
            # Personalized recommendations
            st.markdown("---")
            st.markdown("#### 💡 Personalized Recommendations")
            
            recommendations = []
            
            if delay_tasks in ['Often', 'Always']:
                recommendations.append("🎯 Try breaking large tasks into smaller, manageable chunks")
            if unfinished_tasks in ['Often', 'Always']:
                recommendations.append("✅ Use a task completion checklist to track your progress")
            if fail_routine in ['Often', 'Always']:
                recommendations.append("📅 Start with one simple routine and build from there")
            if physical_activity in ['Not active', 'Slightly active']:
                recommendations.append("🏃 Aim for at least 30 minutes of activity daily")
            if phone_scrolling in ['Often', 'Always']:
                recommendations.append("📱 Set app time limits and create phone-free zones")
            
            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("🎉 Keep up the great work! You're maintaining excellent habits!")
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please make sure all your selections are valid.")

# Sidebar with information
with st.sidebar:
    st.markdown("### ℹ️ About This App")
    st.info("""
    This application uses a **Random Forest Classifier** trained on survey data to predict 
    your activity level based on your behavioral patterns and lifestyle choices.
    
    **Score Guide:**
    - 1️⃣ Very Active & Productive
    - 2️⃣ Active & Engaged  
    - 3️⃣ Moderate Activity
    - 4️⃣ Low Activity Level
    - 5️⃣ Very Low Activity
    """)
    
    st.markdown("### 📈 Model Statistics")
    st.success(f"""
    - **Algorithm:** Random Forest
    - **Training Samples:** {len(df)}
    - **Features:** 5 behavioral indicators
    - **Decision Trees:** 100
    - **Possible Scores:** {len(classes)}
    """)
    
    st.markdown("### 💡 Tips for Accuracy")
    st.warning("""
    - Answer honestly for best results
    - Think about your typical behavior
    - No right or wrong answers
    - Use results for self-improvement
    """)
    
    st.markdown("### 📊 Dataset Info")
    if 'Age' in df.columns:
        st.metric("Age Range", f"{df['Age'].min()} - {df['Age'].max()}")
    st.metric("Total Participants", len(df))