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
from torch.cuda.amp import autocast
import asyncio
import time


#model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
checkpoint_path = os.path.join(os.getcwd(),checkpoint)
# Get the current working directory
start_path = os.getcwd()
# os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '2'
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
        # print(text)
        final_texts = final_texts + text.page_content
    return final_texts

#LLM pipeline
# def llm_pipeline(filepath,min_length,max_length):
#     # max_length assign from streamlit app
#     # Summarization pipeline using T5 model and in Summarization print the summary in asynchrounous way
#     # max_length = st.sidebar.slider("Max Summary Length", min_value=50, max_value=500, value=200)
#     # min_length = st.sidebar.slider("Min Summary Length", min_value=50, max_value=max_length-1, value=50)
#     # st.write("Max Length: ", max_length, "Min Length: ", min_length)
#     pipe_sum = pipeline(
#         'summarization',
#         model=base_model,
#         tokenizer=tokenizer,
#         max_length=max_length,
#         min_length=min_length,
#         use_fast=True
#     )
#     input_text = file_preprocessing(filepath)
#     result = pipe_sum(input_text)
#     result = result[0]['summary_text']
#     return result

# # LLM pipeline with chunk processing
# def llm_pipeline(filepath, min_length, max_length):
#     input_text = file_preprocessing(filepath)
#     # Split the text based on token length
#     inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
#     # Break input into smaller chunks if necessary
#     chunks = [input_text[i:i + 512] for i in range(0, len(input_text), 512)]
    
#     summaries = []
#     for chunk in chunks:
#         inputs = tokenizer(chunk, return_tensors="pt", max_length=512, truncation=True, padding=True)
#         summary_ids = base_model.generate(inputs["input_ids"], max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         summaries.append(summary)
    
#     return " ".join(summaries)

def llm_pipeline(filepath, min_length, max_length, batch_size=4):
    input_text = file_preprocessing(filepath)

    max_chunk_length = 512
    text_chunks = [input_text[i:i + max_chunk_length] for i in range(0, len(input_text), max_chunk_length)]

    summaries = []

    # Enable mixed precision for faster inference
    with autocast():
        for i in range(0, len(text_chunks), batch_size):
            batch_chunks = text_chunks[i:i + batch_size]
            inputs = tokenizer(batch_chunks, return_tensors="pt", padding=True, truncation=True).input_ids.to(base_model.device)
            summary_ids = base_model.generate(inputs, max_length=max_length, min_length=min_length)
            batch_summaries = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids]
            summaries.extend(batch_summaries)
            # st.markdown(batch_summaries)
    # print(summaries)
    final_summary = "\n ".join(summaries)
    return final_summary



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

async def llm_pipeline_async(filepath, min_length, max_length, batch_size=4):
    return await asyncio.to_thread(llm_pipeline, filepath, min_length, max_length, batch_size)


def main():
    st.title("Document Summarization App")
    
    max_length = st.sidebar.slider("Max Summary Length", min_value=50, max_value=1000, value=200)
    min_length = st.sidebar.slider("Min Summary Length", min_value=50, max_value=max_length-1, value=50)


    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])  # Restricting the maximum upload size to 10MB
    if uploaded_file is not None:
        file_size = uploaded_file.size
        if file_size > 2*1024*1024:
            st.error("File size exceeds 2MB")
            return
        
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            filepath = "data/"+uploaded_file.name
            if not os.path.exists("data"):
                os.makedirs("data")
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                pdf_viewer(input=filepath, height=600, width=1000)
            # pdf_viewer(input=filepath, height=600, width=1000)
            # with col1:
            #     st.info("Uploaded File")
            #     pdf_view = displayPDF(filepath)

            with col2:
                with st.spinner("Summarizing..."):
                    summary = asyncio.run(llm_pipeline_async(filepath, min_length, max_length))
                st.success("Summarization Complete!")
                st.info(summary)

                # summary = llm_pipeline(filepath,min_length,max_length)
                # st.success(summary)



if __name__ == "__main__":
    main()
