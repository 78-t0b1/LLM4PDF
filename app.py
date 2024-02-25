import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter 
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://iconape.com/wp-content/png_logo_vector/outreachy-bot-avatar.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn-sdotid.adg.id/images/01a81686-867e-449e-9af4-9f72de33ac7c_1920x1920.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

def getPDFText(docs):
    text = ""
    for doc in docs:
        reader = PdfReader(doc)
        for page in reader.pages:
            text += page.extract_text()
    return text

def getChunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def getVectorStore(chunks):
    embedding = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embedding)
    return vectorstore
    
def getConversation(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512}) 
    memory = ConversationBufferMemory(memory_key='history', return_messages=True)   # Study this
    convo_chain =  ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return convo_chain

def handleQue(que):
    response = st.session_state.convo({'question': que})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config('Chat with PDFs',':book:')
    st.write(css,unsafe_allow_html=True)
    st.header('Chat with PDFs')
    
    if 'convo' not in st.session_state:
        st.session_state.convo = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_que = st.text_input("Ask question about docs:")
    if user_que:
        handleQue(user_que)

    with st.sidebar:
        st.subheader('Your documents')
        documents = st.file_uploader('Please upload PDFs',accept_multiple_files=True,type=['pdf','PDF'])
        if st.button('Process'):
            with st.spinner('Processing.....'):
                #read pdf
                raw_text = getPDFText(documents)
                
                #create chunks
                chunks = getChunks(raw_text)
                # st.write(chunks)
                #Create vector db with embedings.
                vectorstore = getVectorStore(chunks)
                st.write(vectorstore)

                # Create coversation chain
                st.session_state.convo = getConversation(vectorstore)


if __name__ == "__main__":
    main()