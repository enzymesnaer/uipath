import glob
import re
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pyttsx3


def process_pdf(file_path):
    audio_summary = ""
    file = file_path.split("\\")[-1]
    print(f"Processing {file}")
    audio_summary += f"Processing {file}\n"

    loaders = [PyPDFLoader(file_path)]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    texts = text_splitter.split_documents(docs)

    # Embeddings into Faiss vector DB
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    # all-MiniLM-L6-v2
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.9)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True)

    prompt = "Give a short description on what is this document about and then generate short inference or key points from the documents. Include information about any upcoming events."

    chat_history = []
    result = qa_chain({'question': prompt, 'chat_history': chat_history})

    engine = pyttsx3.init()
    rate = engine.setProperty('rate', 150)

    engine.say(f"Processing {file}")

    answer = result['answer']
    print(answer)
    print("-" * 300, end="\n")

    audio_summary += answer + '\n' + '-' * 300 + '\n'

    txt_file_path = f'C:\\Users\\kolisn\\Documents\\Workspace\\hackathon\\MVP\\output\\audio_summary.txt'

    with open(txt_file_path, 'w') as text_file:
        text_file.write(audio_summary)

    engine.say(answer)
    engine.runAndWait()


if __name__ == "__main__":
    pdf_regex = re.compile(r'.*\.pdf$', re.IGNORECASE)
    pdf_files = [file for file in glob.glob('C:\\Users\\kolisn\\Documents\\Workspace\\hackathon\\MVP\\dropbox\\' + '*')
                 if pdf_regex.match(file)]

    for pdf_file in pdf_files:
        process_pdf(pdf_file)
