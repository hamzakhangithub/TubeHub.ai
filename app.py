import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
import yt_dlp
import textwrap
import whisper
import time

import os
#os.environ['PATH'] += os.pathsep + 'ffmpeg\\ffmpeg\\bin\\ffmpeg'
os.environ['OPENAI_API_KEY'] = ''
os.environ["ACTIVELOOP_TOKEN"] = ""

# Define functions for downloading from YouTube, summarizing, and answering queries
llm = ChatOpenAI(model='gpt-4', temperature=0)

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')


def download_from_youtube(url, video_id):
    filename = f'video_{video_id}.mp4'
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': filename,
        'quiet': True,
        'ffmpeg_location':'ffmpeg\\bin'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)

def summarize_video(video_id):
    filename = f'video_{video_id}.mp4'
    model = whisper.load_model('base')
    result = model.transcribe(filename)
    return result['text']

def process_text_for_summarization(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts[:4]]
    docs_as_dicts = [doc.to_dict() for doc in docs]
    return docs_as_dicts

def generate_summary(docs, llm):
    chain = load_summarize_chain(llm, chain_type="refine")
    output_summary = chain.run(docs)
    wrapped_text = textwrap.fill(output_summary, width=100)
    return wrapped_text

def create_qa_chain(retriever, llm):
    prompt_template = """Use the following pieces of transcripts from a video to answer the question in bullet points and summarized. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Summarized answer in bullter points:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=retriever,
                                     chain_type_kwargs=chain_type_kwargs)
    return qa

# Streamlit web app
st.title("TubeHub")

# User input for YouTube URL

docs = []
url = st.text_input("Enter YouTube URL:")
unique_video_id = str(int(time.time()))

if st.button("Summarize"):
    # Download video from YouTube
    download_from_youtube(url, video_id = unique_video_id)

    st.text("Summarizing...")

    progress_bar = st.progress(0)

    # Summarize the video
    summary_text = summarize_video(video_id=unique_video_id)

    # Process text for summarization
    docs_dict = process_text_for_summarization(summary_text)
    
    docs = [Document.from_dict(doc) for doc in docs_dict]

    db = DeepLake(dataset_path="hub://ihamzakhan89/langchain_course_fewshot_selector", embedding_function = embeddings)
# Add documents to the database
    db.add_documents([doc.to_dict() for doc in docs], embedding_data=[doc.page_content for doc in docs])

    
    
    # Generate summary
    summary = generate_summary(docs, llm)

    # Display summary
    st.sidebar.title("Video Summary")
    st.sidebar.text(summary)

    

    progress_bar.progress(100)

# User input for queries
query = st.text_input("Ask a question about the video:")
if st.button("Answer"):
    # Process query
    docs_query = process_text_for_summarization(query)
     
    docs_query_as_dicts = [doc.to_dict() for doc in docs_query]
    docs_query = [Document.from_dict(doc) for doc in docs_query_as_dicts]

    # Create retriever
    db = DeepLake(dataset_path="hub://ihamzakhan89/langchain_course_fewshot_selector", embedding_function = embeddings)

    if docs_query is not None:
        # Check if docs is not None before adding to the database
        # Add documents to the database
        db.add_documents([doc.to_dict() for doc in docs], embedding_data={'texts': [doc.page_content for doc in docs]})

    

        retriever = db.as_retriever()
        retriever.search_kwargs['distance_metric'] = 'cos'
        retriever.search_kwargs['k'] = 4

        # Create QA chain
        qa_chain = create_qa_chain(retriever, llm)

        # Answer query
        answer = qa_chain.run(docs_query)
        st.text(answer)
    else:
        st.warning("No documents to add to the database. Please summarize the video first.")


