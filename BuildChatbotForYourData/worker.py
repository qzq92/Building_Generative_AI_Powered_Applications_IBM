import os
import torch
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain import hub

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None
chat_history = []

def init_llm() -> HuggingFaceEndpoint:
    """Function which returns llm model via HuggingFaceEndpoint (API call)

    Returns:
       HuggingFaceEndpoint: HuggingFaceEndpoint API reference
    """
    # repo name for the model
    model_id = os.environ.get(
        "HUGGINGFACE_QA_LLM_MODEL", 
        default="tiiuae/falcon-7b-instruct"
    )
    # load the model into the HuggingFaceHub
    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        repo_id=model_id,
        model_kwargs={
            "temperature": 0.1,
             "max_new_tokens": 600,
             "max_length": 600}
        )

    return llm

# Function to process a PDF document
def process_document(document_path:str) -> None:
    """Function which loads and chunks the pdf for storage into Chroma vector store. A seperates RetrievalQA chain is constructed upon completion.

    Args:
        document_path (str): Path to pdf document to be processed.
    """

    # Modify conversation_retrieval_chain global variable with RetrievalQA
    global conversation_retrieval_chain

    llm_model = init_llm()
    # Load the document with PyPDF loader. Instantiate with input document and perform load
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    
    # Split the document into chunks using RecursiveCharacterTextSplitter class
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(
            os.environ.get("DOCUMENT_CHUNK_SIZE", default="1024")
        ),
        chunk_overlap=int(
            os.environ.get("DOCUMENT_CHUNK_OVERLAP", default="64")
        ),
    )
    texts = text_splitter.split_documents(documents)

    
    #Initialize embeddings using a pre-trained model to represent the text data.
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=os.environ.get(
            "HUGGINGFACE_DOCUMENT_EMBEDDINGS_MODEL", default="sentence-transformers/all-MiniLM-L6-v2"
        ),
        model_kwargs={"device": DEVICE},
        encode_kwargs = {'normalize_embeddings': False}
    )
    
    # Create an embeddings database using Chroma from the split text chunks.
    db = Chroma.from_documents(texts, embedding=embeddings)

    # Use predefined chat prompt for retrieval qa
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Retrieve more documents with higher diversity
    # Useful if your dataset has many similar documents
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_type="mmr", # maximum marginal relevance (MMR)
            search_kwargs={
                'k': 6, # Limit to 6 Search result
                'lambda_mult': 0.25}
            ),
        return_source_documents=False, # Dont indicate source document as part of return
        input_key = "question",
        chain_type_kwargs={
            "prompt": retrieval_qa_chat_prompt
        }
    )


# Function to process a user prompt
def process_prompt(prompt:str) -> str:
    """Function which processes input prompt as question using conversation retrieval chain and generates a bot response as output taking into account chat history information if available
    """

    #
    global conversation_retrieval_chain
    global chat_history

    # Query the model
    output = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
    answer = output["result"]
    
    # Update the chat history with prompt (user) and bot answer
    chat_history.append((prompt, answer))
    
    # Return the model's response
    return answer