import glob
import json
import os
import re
import sys

import requests
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def insert_line_breaks(input_string, words_per_line=25):
    words = input_string.split()
    result = ""

    for i, word in enumerate(words):
        result += word + " "
        if (i + 1) % words_per_line == 0:
            result += '\n'

    return result.strip()


def process_pdf(file_path, query):
    try:
        loaders = [PyPDFLoader(file_path)]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

        texts = text_splitter.split_documents(docs)

        #========================================================================================

        s = requests.Session()

        api_base = "https://api.endpoints.anyscale.com/v1"
        token =  'esecret_++++bylmr1args2fi4w++++++1dgzvnxk++++++t64'
        url = f"{api_base}/chat/completions"
        body = {
            "model": "meta-llama/Llama-2-70b-chat-hf",
            "messages": [{"role": "system", "content": f"""{texts}"""},
                         {"role": "user", "content": query}],
            "temperature": 0.5
        }

        result = ''
        with s.post(url, headers={"Authorization": f"Bearer {token}"}, json=body) as resp:
            print(resp.json())
            result = json.load(resp.json())
            result = result['choices']['message']['content']

        #========================================================================================

        chat_history = []
        chat_history.append((query, result))

        txt_file_path = f'C:\\MVP\\output\\query_result.txt'

        with open(txt_file_path, 'w') as text_file:
            # Write the content to the file
            text_file.write(result)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    try:
        query = sys.argv[1]
        pdf_regex = re.compile(r'.*\.pdf$', re.IGNORECASE)
        pdf_files = [file for file in glob.glob('C:\\MVP\\dropbox\\' + '*') if pdf_regex.match(file)]

        for pdf_file in pdf_files:
            print(pdf_file)
            process_pdf(pdf_file, query)
    except Exception as e:
        print(e)