import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel , Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")



class ProductInfo(BaseModel):
    product_name: str
    product_details: str
    tentative_price_usd: int= Field(description="Price in USD, must be an integer", gt=0, le=10000,)
# Step 2: Create a prompt template

prompt_template = """
You are a helpful assistant. When asked about any product, respond ONLY in JSON format like:
{{"product_name": "<product name>",
 "product_details": "<short details>",
 "tentative_price_usd": $<integer>}}
"""

prompt=ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        ("user","{input}")
    ]
)




# Sample model list
model_list = ["gemma2-9b-it", "llama3-70b-8192", "meta-llama/Llama-Guard-4-12B"]
# Sample product list
product_list = ["iPhone 16", "Samsung s24", "Samsung s25", "Samsung Ultra","Other"]
# App Title
st.set_page_config(page_title="Price telling App", layout="centered")
st.title("🔍 Price Information Dashboard")


# Step 1: Select Model
selected_model = st.selectbox("Select Groq Model", model_list)

# Step 2: Select or Enter Product
selected_product = st.selectbox("Select a Product", product_list)


custom_product_name = None
if selected_product == "Other":
    custom_product_name = st.text_input("Enter your Product Name:")
    final_product = custom_product_name if custom_product_name else "iPhone 16"
else:
    final_product = selected_product

button=st.button("Fetch Product Details")





if button and selected_model and final_product:
    
    model=ChatGroq(model=selected_model)
    ### chaining
    parser = JsonOutputParser(pydantic_object=ProductInfo)
    chain=prompt|model|parser
    # Step 3: Define a dictionary for product details   
    response=chain.invoke({"input":final_product})
    # Step 4: Create the assistant function
    st.markdown("---")
    st.subheader("🧾 Product Details:")
    st.write(f"**Product_name:** {response['product_name']}")
    st.write(f"**Product Details:** {response['product_details']}")
    st.write(f"**Tentative Price (USD):** ${response['tentative_price_usd']}")
    st.markdown("---")
    st.success("Product details fetched successfully!")