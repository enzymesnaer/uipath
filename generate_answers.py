import glob
import re
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from twilio.rest import Client

account_sid = 'AC03e77649f31d37eb2d902d3e4c906a97'
auth_token = '61462ebc9c022ab5a11f2ded13a0f24b'

def insert_line_breaks(input_string, words_per_line=25):
    words = input_string.split()
    result = ""

    for i, word in enumerate(words):
        result += word + " "
        if (i + 1) % words_per_line == 0:
            result += '\n'

    return result.strip()

def send_whatsapp(output_file):
    client = Client(account_sid, auth_token)
    numbers = ['whatsapp:+918285281211']
    
    with open('output/audio_summary.txt') as f:
        lines=f.readlines()
    file1= ''.join(lines)[:1598]

    for number in numbers:
        message = client.messages.create(
        from_='whatsapp:+14155238886',
        body = file1,
        to   = number
        )

def process_pdf(file_path):
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

    prompt_inputs_path = r'C:\Users\kolisn\Documents\Workspace\hackathon\MVP\prompts.txt'

    with open(prompt_inputs_path, 'r') as file:
        lines = file.readlines()

    chat_history = []
    for itr, prompt in enumerate(lines):
        if itr == 5:
            break
        result = qa_chain({'question': prompt, 'chat_history': chat_history})
        chat_history.append((prompt, result['answer']))
        answer = result['answer']
        print(" " + prompt)
        print(answer + "\n\n")

    file = file_path.split("\\")[-1]
    txt_file_path = f'C:\\Users\\kolisn\\Documents\\Workspace\\hackathon\\MVP\\output\\{file}'.replace('.pdf', '.txt')

    txtcontent = "AI generated answers to F.A.Q's\n\n"

    for key, value in chat_history:
        # Concatenate key and value to the result string
        txtcontent += f'{"Q: " + insert_line_breaks(key)}\n{"A:" + insert_line_breaks(value)}\n\n\n\n'

    with open(txt_file_path, 'w') as text_file:
        # Write the content to the file
        text_file.write(txtcontent)
        send_whatsapp(txt_file_path)



if __name__ == "__main__":

    pdf_regex = re.compile(r'.*\.pdf$', re.IGNORECASE)
    pdf_files = [file for file in glob.glob('C:\\Users\\kolisn\\Documents\\Workspace\\hackathon\\MVP\\dropbox\\' + '*') if pdf_regex.match(file)]

    for pdf_file in pdf_files:
        print(pdf_file)
        process_pdf(pdf_file)
