# import streamlit as st

# # Custom CSS to set the background image
# def set_background_image(image_url):
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("{image_url}");
#             background-size: cover;
#             background-position: center;
#             background-repeat: no-repeat;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # Set the background image using a real URL
# background_image_url = "https://images.unsplash.com/photo-1506744038136-46273834b3fb"
# set_background_image(background_image_url)

# # Your Streamlit app content
# st.title("Welcome to My Streamlit App")
# st.write("This is a Streamlit app with a custom background image.")

import streamlit as st
import base64

# Function to encode the local image file to Base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Custom CSS to set the background image
def set_background_image(image_path):
    encoded_image = get_base64_of_image(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image using a local file path
background_image_path = "./Background Image.png"  # Relative path to the image
set_background_image(background_image_path)

# Your Streamlit app content
st.title("Welcome to My Streamlit App")
st.write("This is a Streamlit app with a custom background image from a local file.")