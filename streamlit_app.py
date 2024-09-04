import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
import os
from streamlit_pdf_viewer import pdf_viewer

#model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
checkpoint_path = os.path.join(os.getcwd(),checkpoint)
# Get the current working directory
start_path = os.getcwd()

# Walk through the directory tree
for root, dirs, files in os.walk(start_path):
    # Print all directories and subdirectories
    print(root)
    for dir in dirs:
        print(os.path.join(root, dir))

print(checkpoint_path)

tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

#file loader and preprocessing
def file_preprocessing(file):
    loader =  PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content
    return final_texts

#LLM pipeline
def llm_pipeline(filepath,min_length,max_length):
    # max_length assign from streamlit app
    # Summarization pipeline using T5 model and in Summarization print the summary in asynchrounous way
    # max_length = st.sidebar.slider("Max Summary Length", min_value=50, max_value=500, value=200)
    # min_length = st.sidebar.slider("Min Summary Length", min_value=50, max_value=max_length-1, value=50)
    # st.write("Max Length: ", max_length, "Min Length: ", min_length)
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=max_length,
        min_length=min_length,
        use_fast=True
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

@st.cache_data
#function to display the PDF of a given file 
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

#streamlit code 
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App")
    
    max_length = st.sidebar.slider("Max Summary Length", min_value=50, max_value=1000, value=200)
    min_length = st.sidebar.slider("Min Summary Length", min_value=50, max_value=max_length-1, value=50)

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            filepath = "data/"+uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                pdf_viewer(input=filepath, height=600, width=1000)
            # pdf_viewer(input=filepath, height=600, width=1000)
            # with col1:
            #     st.info("Uploaded File")
            #     pdf_view = displayPDF(filepath)

            with col2:
                summary = llm_pipeline(filepath,min_length,max_length)
                st.info("Summarization Complete")
                st.success(summary)



if __name__ == "__main__":
    main()
