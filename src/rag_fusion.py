import os
import sys
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.chroma import Chroma
import textwrap

def wrap_text(text, width=90): #preserve_newlines
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

root_dir = sys.path[0][:-4]

class TextPrepper:
    def load_directory(self):
        loader = DirectoryLoader("texts", glob="*.txt", show_progress=True)
        docs = loader.load()
        return docs
    def prepare_texts(self, docs):
        raw_text = ''
        for _, doc in enumerate(docs):
            text = doc.page_content
            if text:
                raw_text += text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap  = 100,
            length_function = len,
            is_separator_regex = False,
        )
        texts = text_splitter.split_text(raw_text)
        return texts

    def prepare_db(self,texts):
        create_db = False # set True to create a new database
        model_name = "BAAI/bge-small-en-v1.5"
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

        embedding_function = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs=encode_kwargs,)
        db_dir = os.path.join(root_dir, "chroma_db")
        if create_db:
        ### Make the chroma and persiste to disk
            db = Chroma.from_texts(texts,
                                embedding_function,
                                persist_directory=db_dir,)
        else:
            db = Chroma(persist_directory=db_dir, embedding_function=embedding_function)
        return db

class RagFusion():
    def __init__(self):
        self.model = "rag-token"
        self.tokenizer = "facebook/rag-token-nq"