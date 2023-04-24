import streamlit as st

from langchain.embeddings.base import Embeddings
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS

class LocalHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_id): 
        self.model = SentenceTransformer(model_id)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using a locally running
           Hugging Face Sentence Transformer model
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        embeddings =self.model.encode(texts)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a locally running HF 
        Sentence trnsformer. 
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        embedding = self.model.encode(text)
        return list(map(float, embedding))

embeddings = LocalHuggingFaceEmbeddings('all-mpnet-base-v2')

vector_store = FAISS.load_local("GillPapers_index", embeddings)

def find_similar_paper_n(title,abstract,n):
    input_text = title + " - " + abstract
    input = embeddings.embed_query(input_text)
    docs = vector_store.similarity_search_by_vector(input, k=n)
    for i in range(0,n):
        meta = docs[i].metadata
        title = meta['TITLE']
        date = meta['Publication_date']
        doi = meta['DOI']
        URL = meta['WARWICK_URL']
        st.write(f'{i+1}.  {title}')
        st.write(f'Publication Date: {date}')
        st.write(f'DOI: {doi}')
        st.write(f'WARWICK URL: {URL}')
        st.write('')

st.set_page_config(page_title='Gillmore', layout='wide')
st.title('Gillmore')
st.markdown(" > Powered by -  Gillmore Centre for Financial Technology ")
# image = Image.open('Sidebar.jpg')
# st.sidebar.image(image)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_title():
    input_title = st.text_input("Title: ","", key="input")
    return input_title

def get_abstract():
    input_abstract = st.text_input("Abstract: "," ")
    return input_abstract

def get_num():
    num = st.number_input("Number of  Outputs: ", min_value=1, max_value=10, value=3, step=1)
    return num

user_title = get_title()
user_abstract = get_abstract()
num = get_num()

# if user_title:
if st.button('Find'):
    find_similar_paper_n(user_title,user_abstract,num)
