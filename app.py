import streamlit as st
import pandas as pd
import numpy as np
import pickle
import replicate
import os
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import base64  # Added import for Base64 decoding

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Fashion Outfit Recommendation and Generation System",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to style the app
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        color: white; /* Ensure text remains white on hover */
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Ensure that your REPLICATE_API_TOKEN is set as an environment variable
if 'REPLICATE_API_TOKEN' not in os.environ:
    st.error("REPLICATE_API_TOKEN not found. Please set it as an environment variable.")
    st.stop()

# Load the saved model
with open('outfit_classification_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the LabelEncoder
with open('label_encoder.pkl', 'rb') as f:
    le_outfit_category = pickle.load(f)

# Load the feature columns
with open('feature_columns.pkl', 'rb') as f:
    model_features = pickle.load(f)

# Load the dataset to get the data for the recommendation
df = pd.read_csv('fashion_dataset.csv')

# If 'Outfit Category' is not in the CSV, recreate it
if 'Outfit Category' not in df.columns:
    # Define the categorize_outfit function
    def categorize_outfit(outfit_description):
        if 'business suit' in outfit_description or 'blazer' in outfit_description or 'pantsuit' in outfit_description:
            return 'Business Attire'
        elif 'evening gown' in outfit_description or 'ball gown' in outfit_description or 'mermaid dress' in outfit_description:
            return 'Evening Wear'
        elif 'sundress' in outfit_description or 'maxi dress' in outfit_description or 'casual dress' in outfit_description:
            return 'Casual Wear'
        elif 'saree' in outfit_description or 'lehenga' in outfit_description or 'salwar kameez' in outfit_description or 'kurti' in outfit_description:
            return 'Traditional Wear'
        elif 'kimono' in outfit_description or 'cheongsam' in outfit_description or 'kaftan' in outfit_description or 'abaya' in outfit_description:
            return 'Cultural Wear'
        else:
            return 'Other'

    # Apply the function to create the 'Outfit Category' column
    df['Outfit Category'] = df['Outfit'].apply(categorize_outfit)

# Prepare the list of options for the dropdown menus
countries = sorted(df['Country'].unique())
events = sorted(df['Event'].unique())
preferred_colors = sorted(df['Preferred Color'].unique())
preferred_styles = sorted(df['Preferred Style'].unique())

# Initialize variables to store suggested outfit and output sentence
if 'suggested_outfit' not in st.session_state:
    st.session_state['suggested_outfit'] = ''
if 'output_sentence' not in st.session_state:
    st.session_state['output_sentence'] = ''
if 'image_url' not in st.session_state:
    st.session_state['image_url'] = ''

# Define the prediction function BEFORE calling it
def predict_outfit(input_data):
    # Convert input_data into DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    # One-hot encode input data
    input_encoded = pd.get_dummies(input_df)

    # Reindex to match training data columns, fill missing columns with 0
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

    # Predict the outfit category
    prediction_encoded = loaded_model.predict(input_encoded)
    predicted_category = le_outfit_category.inverse_transform(prediction_encoded)

    # Filter the original data to find the outfit descriptions in that category
    # Also match other attributes to make the recommendation more accurate
    matching_outfits = df[
        (df['Outfit Category'] == predicted_category[0]) &
        (df['Culture'] == input_data['Culture']) &
        (df['Event'] == input_data['Event']) &
        (df['Preferred Style'] == input_data['Preferred Style'])
    ]

    if not matching_outfits.empty:
        # Choose a random outfit description from the matching outfits
        outfit_description = matching_outfits['Outfit'].sample(1).values[0]
    else:
        outfit_description = "No matching outfit found."

    # Construct the output sentence
    output_sentence = f"THENUSAN, full body image of women wearing {input_data['Preferred Color']} {outfit_description}"

    # Update session state
    st.session_state['suggested_outfit'] = outfit_description
    st.session_state['output_sentence'] = output_sentence

    return outfit_description, output_sentence

# Function to reset fields after downloading
def reset_fields():
    # Reset the session state variables
    st.session_state['selected_country'] = 'Select a country'
    st.session_state['selected_city'] = 'Select a city'
    st.session_state['selected_culture'] = 'Select a culture'
    st.session_state['selected_event'] = 'Select an event'
    st.session_state['selected_color'] = 'Select a color'
    st.session_state['selected_style'] = 'Select a style'
    st.session_state['suggested_outfit'] = ''
    st.session_state['output_sentence'] = ''
    st.session_state['image_url'] = ''

# Main layout with two columns
col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("👗 Recommendation")

    st.write("Please provide the following information to get a personalized outfit suggestion:")

    # Add placeholder options to dropdowns with keys
    country_options = ['Select a country'] + countries
    selected_country = st.selectbox('Country', country_options, key='selected_country')

    if st.session_state['selected_country'] != 'Select a country':
        cities_in_country = sorted(df[df['Country'] == st.session_state['selected_country']]['City'].unique())
    else:
        cities_in_country = []
    city_options = ['Select a city'] + cities_in_country
    selected_city = st.selectbox('City', city_options, key='selected_city')

    if st.session_state['selected_city'] != 'Select a city':
        cultures_in_city = sorted(df[df['City'] == st.session_state['selected_city']]['Culture'].unique())
    else:
        cultures_in_city = []
    culture_options = ['Select a culture'] + cultures_in_city
    selected_culture = st.selectbox('Culture', culture_options, key='selected_culture')

    event_options = ['Select an event'] + events
    selected_event = st.selectbox('Event', event_options, key='selected_event')

    color_options = ['Select a color'] + preferred_colors
    selected_color = st.selectbox('Preferred Color', color_options, key='selected_color')

    style_options = ['Select a style'] + preferred_styles
    selected_style = st.selectbox('Preferred Style', style_options, key='selected_style')

    # When the user clicks the 'Recommend Outfit' button
    if st.button('Recommend Outfit'):
        # Create a dictionary of selections
        selections = {
            'Country': st.session_state['selected_country'],
            'City': st.session_state['selected_city'],
            'Culture': st.session_state['selected_culture'],
            'Event': st.session_state['selected_event'],
            'Preferred Color': st.session_state['selected_color'],
            'Preferred Style': st.session_state['selected_style']
        }

        # Check if all selections have been made
        missing_fields = [field_name for field_name, value in selections.items() if value.startswith('Select')]

        if missing_fields:
            st.error('Please select the following fields before getting a recommendation: ' + ', '.join(missing_fields))
        else:
            # Prepare input data
            input_data = selections

            # Get the outfit prediction and output sentence
            suggested_outfit, output_sentence = predict_outfit(input_data)

            # Display the suggested outfit
            st.success(f"**Suggested Outfit:** {st.session_state['selected_color']} {suggested_outfit}")

    # Display the suggested outfit if it exists in session state
    elif st.session_state.get('suggested_outfit'):
        st.success(f"**Suggested Outfit:** {st.session_state['selected_color']} {st.session_state['suggested_outfit']}")

with col2:
    st.header("🎨 Generation")

    # Only show the 'Generate Image' button if an outfit has been suggested
    if st.session_state['suggested_outfit'] and st.session_state['output_sentence']:
        if st.button('Generate Image'):
            # Show a loading spinner while the image is being generated
            with st.spinner('Generating image...'):
                try:
                    # Define the input parameters for the model
                    inputs = {
                        "model": "dev",
                        "prompt": st.session_state['output_sentence'],
                        "lora_scale": 1,
                        "num_outputs": 1,
                        "aspect_ratio": "1:1",
                        "output_format": "webp",
                        "guidance_scale": 3.5,
                        "output_quality": 90,
                        "prompt_strength": 0.8,
                        "extra_lora_scale": 1,
                        "num_inference_steps": 28
                    }

                    # Run the model using replicate.run()
                    output = replicate.run(
                        "thenuri/flux-full-body-image-generation:30176a44e38cddb6dd3adb8f191f6da2f6ff5cce5dc837201855c8d2a6c95e24",
                        input=inputs
                    )

                    # The output is a list of FileOutput objects
                    if output and isinstance(output, list) and len(output) > 0:
                        image_output = output[0]
                        image_url = image_output.url  # Extract the URL string

                        # Update session state
                        st.session_state['image_url'] = image_url

                        # Check if the image URL is a data URL
                        if image_url.startswith('data:'):
                            # Decode the Base64 image data
                            header, encoded = image_url.split(',', 1)
                            image_data = base64.b64decode(encoded)
                            st.image(image_data, caption='Generated Outfit Image')

                            # Provide a download button with on_click callback
                            st.download_button(
                                label="Download Image",
                                data=image_data,
                                file_name='generated_outfit.webp',
                                mime='image/webp',
                                on_click=reset_fields  # Reset fields after download
                            )
                        else:
                            # Handle the case where 'image_url' is a regular URL
                            image_response = requests.get(image_url)
                            if image_response.status_code == 200:
                                image_data = image_response.content
                                st.image(image_data, caption='Generated Outfit Image')

                                # Provide a download button with on_click callback
                                st.download_button(
                                    label="Download Image",
                                    data=image_data,
                                    file_name='generated_outfit.webp',
                                    mime='image/webp',
                                    on_click=reset_fields  # Reset fields after download
                                )
                            else:
                                st.error("Failed to retrieve the generated image for downloading.")
                    else:
                        st.error("No valid output returned from the API.")
                except Exception as e:
                    st.error(f"An error occurred during image generation: {e}")
                    import traceback
                    st.text(traceback.format_exc())
        # Display the generated image if it exists in session state
        elif st.session_state.get('image_url'):
            image_url = st.session_state['image_url']
            if image_url.startswith('data:'):
                header, encoded = image_url.split(',', 1)
                image_data = base64.b64decode(encoded)
                st.image(image_data, caption='Generated Outfit Image')
            else:
                st.image(image_url, caption='Generated Outfit Image')
    else:
        st.info("Please get a recommendation first to generate an image.")
