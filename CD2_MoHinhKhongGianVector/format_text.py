# N19DCCN104 - Trần Thị Trúc Ly
# N19DCCN097 - Nguyễn Thị Mỹ Linh
# N19DCCN024 - Nguyễn Bảo Chính


def load_and_format(path_docs):
    # Đọc file
    with open(path_docs, 'r') as f:
        documents = f.read()

    # tách từng văn bản
    docID_documents = documents.lower().replace("\n"," ").split("/")

    #Bỏ phần tử cuối sau khi split(vì cuối file doc-text là / nên  phần tử cuối sau khi split là chuỗi rỗng '')
    docID_documents.pop()

    # tách ID và văn bản
    documents = []
    for document in docID_documents:
        # cắt khoảng trống đầu và cuối
        document = document.strip()

        # Tìm vị trí khoảng trắng giữa id và văn bản để tách lấy nội dung
        index = document.find(" ")

        # add nội dung của từng văn bản
        documents.append(document[index+1:])

    # Trả về mảng chứa nội dung của các văn bản
    return documents

    