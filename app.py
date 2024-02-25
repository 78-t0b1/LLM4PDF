import streamlit as st


def main():
    st.set_page_config('Chat with PDFs',':book:')
    st.header('Chat with PDFs')
    st.text_input('Ask question here')

    with st.sidebar:
        st.subheader('Your documents')
        st.file_uploader('Please upload PDFs')
        st.button('Process')

if __name__ == "__main__":
    main()