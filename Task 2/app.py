import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Movie Rating Predictor", page_icon="🎬", layout="wide")

st.title("🎬 IMDb Movie Rating Predictor")
st.write("Predict IMDb movie ratings based on movie information!")

model = None
encoders = None

try:
    with open('movie_model.pkl', 'rb') as file:
        model = pickle.load(file)
except:
    st.error("Error: Could not load movie_model.pkl")
    st.stop()

try:
    with open('target_encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
except:
    st.error("Error: Could not load target_encoders.pkl")
    st.stop()

st.write("---")
st.subheader("Enter Movie Details:")

col1, col2, col3 = st.columns(3)

with col1:
    year = st.slider("Movie Year", 1950, 2024, 2020)

with col2:
    duration = st.number_input("Duration (minutes)", 30, 300, 120)

with col3:
    votes = st.number_input("Number of Votes", 1, 1000000, 10000)

st.write("---")
st.write("### Additional Info:")

col4, col5, col6 = st.columns(3)

with col4:
    genre = st.text_input("Genre", placeholder="e.g., Action")

with col5:
    director = st.text_input("Director", placeholder="e.g., Aamir Khan")

with col6:
    actor1 = st.text_input("Actor 1", placeholder="e.g., Shah Rukh Khan")

col7, col8 = st.columns(2)

with col7:
    actor2 = st.text_input("Actor 2", placeholder="e.g., Deepika")

with col8:
    actor3 = st.text_input("Actor 3", placeholder="e.g., Amitabh")
st.write("---")

if st.button("Predict Rating", use_container_width=True):
    if not genre or not director or not actor1:
        st.warning("Please fill in Genre, Director, and Actor 1")
    else:
        def encode_value(value, feature_type):
            if feature_type in encoders:
                if value in encoders[feature_type]:
                    return encoders[feature_type][value]
                else:
                    st.warning(f"'{value}' not found in training data. Using default value.")
                    return 5.8
            else:
                return 5.8
        
        genre_encoded = encode_value(genre, 'Genre')
        director_encoded = encode_value(director, 'Director')
        actor1_encoded = encode_value(actor1, 'Actor 1')
        
        if actor2:
            actor2_encoded = encode_value(actor2, 'Actor 2')
        else:
            actor2_encoded = 5.8
        
        if actor3:
            actor3_encoded = encode_value(actor3, 'Actor 3')
        else:
            actor3_encoded = 5.8
        
        features = np.array([[year, duration, votes, genre_encoded, director_encoded, actor1_encoded, actor2_encoded, actor3_encoded]])
        
        try:
            prediction = model.predict(features)[0]
            
            if prediction > 10:
                prediction = 10.0
            if prediction < 1:
                prediction = 1.0
            
            st.success("Prediction Complete!")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.metric("Predicted Rating", f"{prediction:.2f}/10")
            
            with col_res2:
                if prediction >= 8:
                    cat = "Outstanding! 🚀"
                elif prediction >= 7:
                    cat = "Excellent 😊"
                elif prediction >= 6:
                    cat = "Good 👍"
                elif prediction >= 5:
                    cat = "Decent 👌"
                else:
                    cat = "Below Average 👎"
                st.metric("Category", cat)
            
            st.info(f"""
            Movie Details:
            - Year: {year}
            - Duration: {duration} min
            - Votes: {votes:,}
            - Genre: {genre}
            - Director: {director}
            - Actors: {actor1}, {actor2 if actor2 else "N/A"}, {actor3 if actor3 else "N/A"}
            
            **Predicted Rating: {prediction:.2f}**
            """)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

with st.sidebar:
    st.header("About")
    st.write("This app predicts IMDb movie ratings using machine learning!")
    
    st.header("How it works")
    st.write("1. Enter movie details")
    st.write("2. Click Predict")
    st.write("3. Get your rating prediction")
    
    st.header("Tips")
    st.write("- Use exact names from the dataset")
    st.write("- Unknown actors use default value (5.8)")
    st.write("- More votes = higher ratings")
