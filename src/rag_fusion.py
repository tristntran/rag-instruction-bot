import os
import sys
from langchain.llms import GPT4All
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import textwrap
from langchain.load import dumps, loads

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
    def __init__(self) -> None:
        self.create_db = False

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

    def prepare_db(self):
        model_name = "BAAI/bge-small-en-v1.5"
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

        embedding_function = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs=encode_kwargs,)
        db_dir = os.path.join(root_dir, "chroma_db")
        if self.create_db:
        ### Make the chroma and persiste to disk
            texts = self.prepare_texts(self.load_directory())
            db = Chroma.from_texts(texts,
                                embedding_function,
                                persist_directory=db_dir,)
        else:
            db = Chroma(persist_directory=db_dir, embedding_function=embedding_function)
        return db

class RagFusion():
    def __init__(self):
        self.model = self.load_model()
        self.tokenizer = "facebook/rag-token-nq"
        self.retriever = self.get_retriever()
        self.chain = self.create_rag_chain()
    
    def load_model(self):
        model_path = os.path.join(root_dir,
                          "model",
                          "mistral-7b-instruct-v0.1.Q4_0.gguf")

        model = GPT4All(model=model_path)
        return model
    def get_retriever(self):
        prepper = TextPrepper()
        db = prepper.prepare_db()
        return db.as_retriever(k=5, fetch_k=20, search_type="mmr")

    def create_rag_chain(self):
        query_prompt = ChatPromptTemplate(
                input_variables=["original_query"],
                messages=[
                    SystemMessagePromptTemplate(
                        prompt=PromptTemplate(
                            input_variables=[],
                            template="You are a helpful assistant that generates multiple search queries based on a single input query.",
                        )
                    ),
                    HumanMessagePromptTemplate(
                        prompt=PromptTemplate(
                            input_variables=["original_query"],
                            template="Generate multiple search queries related to: {question} \n OUTPUT (4 queries):",
                        )
                    ),
                ],
            )
        generate_queries = (
            query_prompt | self.model | StrOutputParser() | (lambda x: x.split("\n"))
        )
        ragfusion_chain = generate_queries | self.retriever.map() | self.reciprocal_rank_fusion
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        full_rag_fusion_chain = (
            {
                "context": ragfusion_chain,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.model
            | StrOutputParser()
        )
        return full_rag_fusion_chain

    def reciprocal_rank_fusion(self, results: list[list], k=60):
        fused_scores = {}
        for docs in results:
            # Assumes the docs are returned in sorted order of relevance
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                previous_score = fused_scores[doc_str]
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return reranked_results