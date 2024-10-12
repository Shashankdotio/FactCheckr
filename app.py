import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from factcheckr import manual_testing  # Importing from factcheckr.py

# Styling with custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
        background-color: #E9F1F5;  /* Light background color */
    }
    
    .title {
        color: #007BFF;  /* Darker blue for title */
        text-align: center;
        font-weight: 700;
        font-size: 3.5em;
        margin-bottom: 0px;
    }
    
    .subtitle {
        color: #606C76;  /* Gray for subtitle */
        text-align: center;
        font-size: 1.5em;
        font-weight: 300;
        margin-top: 0px;
    }
    
    .stButton>button {
        background-color: #007BFF;  /* Blue button */
        color: white;
        font-weight: 500;
        border-radius: 25px;  /* Rounded button */
        font-size: 1.1em;
        transition: background-color 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #0056b3;  /* Darker blue on hover */
    }
    
    hr {
        border: none;
        height: 3px;
        background-color: #007BFF;
    }
    
    .footer {
        text-align: center;
        font-size: small;
        color: #888;
    }

    .metric {
        font-size: 1.5em;
        font-weight: 700;
    }

    .progress-bar {
        height: 20px;
        border-radius: 10px;  /* Curved edges for progress bar */
        background-color: #E0E0E0;
        overflow: hidden;
        margin-bottom: 10px;
    }
    
    .progress-fill-fake {
        height: 100%;
        transition: width 0.8s ease;  /* Smooth transition for filling */
        border-radius: 10px;  /* Curved edges for progress fill */
        background-color: #FF4B4B;  /* Red for Fake */
        width: 0;  /* Initial width */
    }
    
    .progress-fill-real {
        height: 100%;
        transition: width 0.8s ease;  /* Smooth transition for filling */
        border-radius: 10px;  /* Curved edges for progress fill */
        background-color: #4CAF50;  /* Green for Real */
        width: 0;  /* Initial width */
    }

    .divider {
        height: 2px;
        background-color: #E0E0E0;  /* Light gray for dividers */
        margin: 15px 0;
        border-radius: 5px;  /* Rounded edges for dividers */
    }

    .average-section {
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);  /* Subtle shadow effect */
    }

    .average-title {
        color: #007BFF;  /* Title color for averages */
        font-size: 2em;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .model-name {
        color: #007BFF;  /* Different color for model names */
        font-size: 1.2em;  /* Slightly larger font size */
        font-weight: 500;  /* Medium weight */
        text-align: center;  /* Center alignment */
    }
    </style>
    """, unsafe_allow_html=True)

# App Header with Title, Subtitle, and Logo
st.image("logo.png", use_column_width=True)  # Display the logo
st.markdown("<h1 class='title'>FactCheckr™</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subtitle'>Is it Real or Fake? Let AI Decide!</h3>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar with App Description and Instructions
st.sidebar.title("FactCheckr™")
st.sidebar.markdown("A fake news detection web app.")
st.sidebar.markdown("**Instructions:**")
st.sidebar.markdown("""
1. Enter the news text you want to classify.
2. Click on **Classify** to get predictions from various models.
""")

# Input Section
st.markdown("### Enter News Article Text:")
news = st.text_area("Paste news text here:", placeholder="Enter the news article text you want to analyze.")

# Classification Button and Output Section
if st.button("Classify"):
    if news.strip() != "":
        predictions = manual_testing(news)
        st.write("### Model Predictions:")
        
        # Variables to calculate averages and store results
        total_fake = 0
        total_real = 0
        model_count = len(predictions)
        
        model_names = []
        fake_percentages = []
        real_percentages = []

        # Display results with dividers
        for model_name, result in predictions.items():
            model_names.append(model_name)
            fake_percentage = result['Fake']
            real_percentage = result['Real']
            fake_percentages.append(fake_percentage)
            real_percentages.append(real_percentage)

            # Update totals for average calculation
            total_fake += fake_percentage
            total_real += real_percentage

            # Create columns for side-by-side progress bars
            col1, col2 = st.columns(2)  # Create two columns

            # Display the model name in a different color
            st.markdown(f"<p class='model-name'>{model_name}</p>", unsafe_allow_html=True)  # Model name displayed above

            # Fake Progress Bar in the first column
            with col1:
                st.markdown('<div class="progress-bar"><div class="progress-fill-fake" style="width: {}%;"></div></div>'.format(fake_percentage), unsafe_allow_html=True)
                st.markdown(f"**Fake:** {fake_percentage:.2f}%", unsafe_allow_html=True)

            # Real Progress Bar in the second column
            with col2:
                st.markdown('<div class="progress-bar"><div class="progress-fill-real" style="width: {}%;"></div></div>'.format(real_percentage), unsafe_allow_html=True)
                st.markdown(f"**Real:** {real_percentage:.2f}%", unsafe_allow_html=True)
            
            # Add a horizontal divider between results
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Calculate and display averages
        average_fake = total_fake / model_count
        average_real = total_real / model_count

        # Average Predictions Section
        st.markdown("<h3 class='average-title'>Average Predictions:</h3>", unsafe_allow_html=True)

        # Create columns for average progress bars
        avg_col1, avg_col2 = st.columns(2)  # Create two columns for averages

        # Average Progress Bar for Fake
        with avg_col1:
            st.markdown('<div class="progress-bar"><div class="progress-fill-fake" style="width: {}%;"></div></div>'.format(average_fake), unsafe_allow_html=True)
            st.markdown(f"**Average Fake:** {average_fake:.2f}%", unsafe_allow_html=True)

        # Average Progress Bar for Real
        with avg_col2:
            st.markdown('<div class="progress-bar"><div class="progress-fill-real" style="width: {}%;"></div></div>'.format(average_real), unsafe_allow_html=True)
            st.markdown(f"**Average Real:** {average_real:.2f}%", unsafe_allow_html=True)

        # Visualization Section
        st.markdown("### Visualization of Predictions:")
        # Create a DataFrame for visualization
        df = pd.DataFrame({
            'Model': model_names,
            'Fake': fake_percentages,
            'Real': real_percentages
        })

        # Set the bar width and position
        bar_width = 0.35
        x = np.arange(len(model_names))

        # Create the bar chart
        fig, ax = plt.subplots()
        ax.bar(x - bar_width/2, df['Fake'], bar_width, label='Fake', color='red')
        ax.bar(x + bar_width/2, df['Real'], bar_width, label='Real', color='green')

        # Labeling the chart
        ax.set_xlabel('Models')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Real vs Fake Predictions')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Display the chart in Streamlit
        st.pyplot(fig)

    else:
        st.write("Please enter some text to classify.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class='footer'>
    Made by Shashank Kamble - LinkedIn: shashankkamble97 | GitHub: Shashankdotio
</div>
""", unsafe_allow_html=True)
