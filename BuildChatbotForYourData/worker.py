import os
import torch
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables to be updated
conversation_retrieval_chain = None
chat_history = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key='answer',
            return_messages=True
        )

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
    # load the model into the HuggingFaceHub, set token limit
    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        repo_id=model_id,
        temperature=0.1,
        max_new_tokens=600,
    )
    return llm

# Function to process a PDF document
def process_document(document_path:str) -> None:
    """Function which loads and chunks the pdf for storage into Chroma vector store. A seperates RetrievalQA chain is constructed upon completion without any return.

    Args:
        document_path (str): Path to pdf document to be processed.
    """

    # Modify conversation_retrieval_chain global variable with RetrievalQA
    global conversation_retrieval_chain

    llm_model = init_llm()
    # Load the document with PyPDF loader. Instantiate with input document and perform load
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    
    # Split the document into chunks using RecursiveCharacterTextSplitter class by instantiating and calling split_documents.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(
            os.environ.get("DOCUMENT_CHUNK_SIZE", default="1024")
        ),
        chunk_overlap=int(
            os.environ.get("DOCUMENT_CHUNK_OVERLAP", default="64")
        ),
    )
    texts = text_splitter.split_documents(documents)

    # Initialize embeddings using a pre-trained model to represent the text data.
    embeddings =  HuggingFaceEmbeddings(
        model_name=os.environ.get(
            "HUGGINGFACE_DOCUMENT_EMBEDDINGS_MODEL", 
            default="tiiuae/falcon-7b-instruct",
        ),
        model_kwargs={"device": DEVICE},
        encode_kwargs = {'normalize_embeddings': False}
    )
    
    # Create an embeddings database using Chroma from the split text chunks.
    db = Chroma.from_documents(texts, embedding=embeddings)

    # Use predefined chat prompt template for retrieval qa

    custom_template = """Use the following pieces of context and also chat history to answer the question at the end. If you don't know the answer, just say I do not know and do not attempt to make up an answer. The answer should be a summary, limited to two sentences.

    Context: {context}
    
    Chat History: {chat_history}
    
    Question: {question}
    Answer:"""
    # Define prompt template for use as chain
    custom_prompt_template = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=custom_template
    )


    # Instantiate ConversationalRetrievalChain that allows chat history to be 
    conversation_retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        chain_type="stuff", # inserts docs all into a prompt,
        retriever=db.as_retriever(
            search_type="mmr", # maximum marginal relevance (MMR)
            search_kwargs={
                'k': 3, # Limit to search result
            }
        ),
        return_source_documents=True, # To validate source of info
        combine_docs_chain_kwargs={'prompt': custom_prompt_template},
        verbose=True,
        memory=chat_history
    )

# Function to process a user prompt
def process_prompt(prompt:str) -> str:
    """Function which processes input prompt as question using conversation retrieval chain and generates a bot response as output taking into account chat history information if available.

    Args:
        document_path (str): Path to pdf document to be processed.

    Returns:
       HuggingFaceEndpoint: HuggingFaceEndpoint API reference
    """

    global conversation_retrieval_chain
    global chat_history

    # Chain input must fill chat prompt template variables. Based on langchain-ai/retrieval-qa-chat, we need to provide input and chat_history when using the chain.
    output_dict = conversation_retrieval_chain(
        {
            "question": prompt,
        }
    )
    print(output_dict)
    answer = output_dict["answer"]
    
    # Return the model's response
    return answer