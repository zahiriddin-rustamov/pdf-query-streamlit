import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

#questions = []
#answers = []

if 'questions' not in st.session_state:
    st.session_state['questions'] = []

if 'answers' not in st.session_state:
    st.session_state['answers'] = []

if 'texts' not in st.session_state:
    st.session_state['texts'] = ''

if 'current_pdf' not in st.session_state:
    st.session_state['current_pdf'] = ''

questions = st.session_state['questions']
answers = st.session_state['answers']

st.title("Query PDFs using LLMs")

with st.sidebar:
    open_ai_key = st.text_input("Enter OpenAI Key")
    os.environ["OPENAI_API_KEY"] = open_ai_key

    try:
        embeddings = OpenAIEmbeddings()
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
    except:
        st.warning("[Please enter OpenAI Key](https://platform.openai.com/account/api-keys)")
    
    #st.header("Model")
    #model = st.selectbox("Select model", ["text-davinci-003", "google"])

    pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, key="pdfs")

    #for pdf in pdfs:
        #st.write(pdf)

    pdf_names = [pdf.name for pdf in pdfs]

    st.caption("---\nThe app only supports OpenAI's model for now. Expect other models to be added soon that do not require access keys.")

#st.multiselect(label="Available PDFs:", options=pdf_names, default=pdf_names)
#current_pdf = ''
#st.write(current_pdf)

st.write("---\n")
selected_pdf = st.radio(label="**Available PDFs:**", options=pdf_names)
st.write("\n")

st.write('---')
question = st.text_input('**Enter your question:**', '', key="question")

run_button = st.button("Run")

if (len(pdfs) > 0) & (len(question) > 5) & (run_button):
    pdf_to_run = [pdf for pdf in pdfs if pdf.name == selected_pdf][0]
    progress_bar = st.progress(0)

    if st.session_state['current_pdf'] != pdf_to_run.name:
        reader = PdfReader(pdf_to_run)
        st.session_state['current_pdf'] = pdf_to_run.name

        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        progress_bar.progress(25)

        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)
        st.session_state['texts'] = texts
        progress_bar.progress(50)
    else:
        texts = st.session_state['texts']

    progress_bar.progress(60)

    try:
        docsearch = FAISS.from_texts(texts, embeddings)
        docs = docsearch.similarity_search(question)
        progress_bar.progress(90)
        answer = chain.run(input_documents=docs, question=question)
        st.write(answer)

        questions.append(question)
        answers.append(answer)

        st.session_state['questions'] = questions
        st.session_state['answers'] = answers
        progress_bar.progress(100)
    except Exception as e:
        st.error(f"{e}")
    
    progress_bar.empty()

container = st.container()
container.write("---")
questions = st.session_state['questions']
answers = st.session_state['answers']

for index, question in enumerate(questions):
    #if index % 2 == 0:
    container.info(f"**{question}**\n\n{answers[index]}")
    #else:
    #    container.success(f"**{question}**\n\n{answers[index]}")
    #container.success(f"\n\n{answers[index]}")
    #html_string = f"<small style='color: rgb(163, 168, 184);font-size:10px;margin:0 !important;padding:0 !important'>{pdfs[index].name}</small>"
    #container.markdown(html_string, unsafe_allow_html=True)
container.write("---")