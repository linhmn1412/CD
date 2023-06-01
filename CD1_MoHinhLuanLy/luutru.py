# N19DCCN104 - Trần Thị Trúc Ly
# N19DCCN097 - Nguyễn Thị Mỹ Linh
# N19DCCN024 - Nguyễn Bảo Chính
import numpy as np
import format_text

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

if __name__ == '__main__':
    path_docs = "./doc-text"
    documents = format_text.format(path_docs)

    # thẻ định vị
    docID = []
    for i, doc in enumerate(documents):
        for word in doc.split():
            # bỏ những từ có trong stopword
            if word not in stop_words:
                docID.append([word, i + 1])

    # sắp xếp theo alphabet
    docID= sorted(docID, key = lambda x: x[0])

    dictionary = {}
    # Trả về danh sách document chứa các từ
    for row in docID: 
        if row[0] in dictionary.keys(): 
            if row[1] not in dictionary[row[0]]:
                dictionary[row[0]].append(row[1])
        else:
            dictionary[row[0]] = [row[1]]

    np.save("./inverted_index.npy", dictionary, allow_pickle=True)

