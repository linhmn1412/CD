# N19DCCN104 - Trần Thị Trúc Ly
# N19DCCN097 - Nguyễn Thị Mỹ Linh
# N19DCCN024 - Nguyễn Bảo Chính

# tiền xử lí file đầu vào doc-text.txt
def format(path_docs):
    # Đọc file doc_texts
    with open(path_docs, 'r') as f:
        documents = f.read()

    # tách từng docID_doc
    docID_documents = documents.lower().replace("\n"," ").split("/")
    docID_documents.pop()

    # tách docID và doc
    documents = []
    for document in docID_documents:
        # cắt khoảng trống đầu và cuối
        document = document.strip()

        # Lấy nội dung của từng document
        index = document.find(" ")
        documents.append(document[index+1:])

    return documents