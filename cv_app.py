import pandas as pd
from transformers import ViTModel
import streamlit as st
from streamlit_theme import st_theme

import base64, time, io
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.callbacks.manager import get_openai_callback
model = ViTModel.from_pretrained

with st.spinner("Preparing Application"):
    theme_json = st_theme()
    time.sleep(1)
    theme = theme_json['base']

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

if theme == "dark":
    background_image_path = "assets/dark_bg.png"
    header_image_path = "assets/dark_wp.png"
else:
    background_image_path = "assets/light_bg.png"
    header_image_path = "assets/light_wp.png"

set_background_image(background_image_path)
st.title("📄 AI Recruitment App")
st.image(header_image_path)

homepage_text = '''**Revolutionize Your Hiring Strategy with AI-Powered CV Screening!**  

Supercharge your recruitment process with our **AI-Powered CV Screening Tool**, engineered to help HR teams swiftly and accurately pinpoint top-tier talent.  
Leveraging cutting-edge **artificial intelligence and machine learning**, our system instantly scans resumes, assesses qualifications, and ranks candidates based on your ideal criteria. 

✅ **Boost Efficiency & Reduce Bias** – Automate CV screening and focus on high-potential candidates.  
✅ **Precision Matching** – Effortlessly align candidate expertise with your role requirements.  
✅ **Hassle-Free Integration** – Sync seamlessly with your existing ATS for a frictionless hiring journey. 

Elevate your HR team with intelligent automation and make confident, data-backed hiring decisions-faster than ever!'''

with st.container(border=True):
    st.markdown(homepage_text)

def chat(req_content, uploaded_cv):
    file_content = uploaded_cv.read()
    mime_type = uploaded_cv.type

    fix_prompt = f'''You are an Expert Recruiters. Your task is review candidate data wether It's match for job requirements or not with given candidate data provided.
Here is job requirements:
{req_content}
'''

    prompt_text = HumanMessage(
        content=[
            {"type": "text", "text": fix_prompt},
            {
                "type": "media",
                "mime_type": mime_type,
                "data": file_content
            },
        ]
    )

    class ResponseFormatter(BaseModel):
        score: int = Field(description="Give score from 0 to 100 for how much this candidate suits for job role, try to consistently give score from 0 to 100")
        reason: str = Field(description="Give the reason about match or not the candidate with needed role")
        desc: str = Field(description="Describe the candidate's skills and capability for needed role")
        suggestion: str = Field(description="Give suggestion for this candidate to improve their skills, or what they need to learn to be better fit for this role")
        contact: str = Field(description="Give the contact information of this candidate, if available")

    model_with_structure = llm.with_structured_output(ResponseFormatter)

    with get_openai_callback() as cb:
        structured_response = model_with_structure.invoke([prompt_text])
        completion_tokens = cb.completion_tokens
        prompt_tokens = cb.prompt_tokens
        score = structured_response.score
        reason = structured_response.reason
        desc = structured_response.desc
        suggestion = structured_response.suggestion
        contact = structured_response.contact

    response = {
        "score" : score,
        "reason" : reason,
        "desc" : desc,
        "suggestion" : suggestion,
        "contact" : contact,
        "completion_tokens" : completion_tokens,
        "prompt_tokens" : prompt_tokens
    }
    return response

api_key = st.text_input("Enter your API Key:", type="password")

if api_key:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp-image-generation",
        api_key=api_key
    )


uploaded_req = st.file_uploader(
    "**Drop Job Requirements Here**", type="txt", accept_multiple_files=False
)

if uploaded_req:
    req_content = uploaded_req.read().decode("utf-8")
    with st.expander("Job Requirements Detail"):
        st.markdown(req_content)
    
    uploaded_cvs = st.file_uploader(
        "**Upload PDF CV**", type="pdf", accept_multiple_files=True
    )

    if uploaded_cvs:
        if st.button("Analyze"):
            st.write("Candidate Analysis Results:")

            result_list = []
            for uploaded_cv in uploaded_cvs:
                st.subheader(f"📘 {uploaded_cv.name}")
                result = chat(req_content, uploaded_cv)
                result['filename'] = uploaded_cv.name
                result_list.append(result)
                st.write(result)
            
            data = pd.DataFrame(
                result_list, columns=["score", "reason", "desc", "suggestion", "filename", "contact"]
            )            

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                data.to_excel(writer, index=False, sheet_name="Sheet1")
                writer.close()

            excel_data = output.getvalue()

            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name="data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
