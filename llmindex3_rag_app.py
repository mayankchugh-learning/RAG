import streamlit as st
import torch
from transformers import GPT3LMHeadModel, GPT2Tokenizer

# Load the LLMINDEX RAG model
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT3LMHeadModel.from_pretrained(model_name)

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the maximum length for generated responses
MAX_LENGTH = 100

# Streamlit app
st.title("LLMINDEX RAG")

# Input text
input_text = st.text_area("Enter your query:", height=100)

# Generate response
if st.button("Generate"):
    st.write("Generating response...")
    
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Generate response
    output = model.generate(input_ids=input_ids, max_length=MAX_LENGTH, num_return_sequences=1)
    
    # Decode and display response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write(response)