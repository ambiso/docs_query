import os
import json

from pprint import pprint
from hashlib import sha256
from multiprocessing import Pool

import pypdf
import tiktoken

from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm

def print_color(text, color):
    print("\033[38;5;{}m{}\033[0m".format(color, text))

def filetree(folder):
    return [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(folder)) for f in fn]

def process(arg):
    (file, existing_hashes) = arg
    if file.endswith(".pdf"):
        #print_color("loading " + file, 34)
        try:
            with open(file, "rb") as f:
                h = sha256()
                while True:
                    data = f.read(1<<12)
                    if not data:
                        break
                    h.update(data)
                h = h.hexdigest()
            if h in existing_hashes or os.path.getmtime(file) < 1683576562:
                print(f"Skipping {file}")
                return ([], h)
            loader = PyPDFLoader(file)
            documents = loader.load_and_split()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
            return (text_splitter.split_documents(documents), h)
        except pypdf.errors.PdfReadError:
            print_color(f"Failed loading {file}", 196)
    return ([], None)


def load_documents(folder, existing_hashes):
    with Pool() as p:
        ft = filetree(folder)
        results = list(tqdm(p.imap(process, ((x, existing_hashes) for x in ft)), total=len(ft)))
        existing_hashes = existing_hashes.union([h for (_, h) in results if h is not None])
        documents = [ x for (y, _) in results for x in y ]
        e = tiktoken.encoding_for_model("gpt-3.5-turbo")
        t = sum(len(e.encode(text.page_content)) for text in documents)
        print(f"Tokens: {t}, Estimated cost: {t/1000*0.0004}")
        return documents, existing_hashes

EX_HS = "existing_hashes.json"

def main():
    embeddings = OpenAIEmbeddings()
    persist_dir = "docs.db"
    # db = Chroma.from_documents(load_documents("/zotero_papers"), embeddings, persist_directory=persist_dir)
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    if not os.path.exists(EX_HS):
        with open(EX_HS, "w") as f:
            f.write(json.dumps([]))
    with open(EX_HS, "r") as f:
        existing_hashes = set(json.load(f))
    (documents, existing_hashes) = load_documents("~/Zotero/storage", existing_hashes)
    print(f"Total documents: {len(existing_hashes)}")
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    if len(texts) > 0:
        db.add_texts(texts=texts, metadatas=metadatas)
    with open(EX_HS, "w") as f:
        f.write(json.dumps(list(existing_hashes)))
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
    while True:
        question = input("Question: ")
        if question == 's':
            print_color("Source documents:", 46)
            pprint(result["source_documents"])
        else:
            result = qa({"query": question})
            print_color("Answer: " + result["result"], 46)

if __name__ == '__main__':
    main()
