# N19DCCN104 - Trần Thị Trúc Ly
# N19DCCN097 - Nguyễn Thị Mỹ Linh
# N19DCCN024 - Nguyễn Bảo Chính

import numpy as np
from collections import Counter
from nltk.corpus import stopwords
import format_text

stop_words = set(stopwords.words('english'))

def remove_stopwords(list):
    list_words_docs = []
    for i, doc in enumerate(list):
        list_words_docs.append([])
        for word in doc.split():
            # bỏ những từ có trong stopword
            if word not in stop_words:
                list_words_docs[i].append(word)
    return list_words_docs

def computeTFIDF(list_words_docs):  
    # Tạo tập từ điển
    vocabs = list(set.union(*map(set, list_words_docs)))
    
    # Tính tf(t,d): Đếm số lần từ t xuất hiện trong văn bản d
    counters = list(map(Counter, list_words_docs))

    # Tạo ma trận zero biểu diễn w_tf
    matrix = np.zeros((len(list_words_docs), len(vocabs)))

    # Duyệt danh sách counters để tính w_tf 
    for i, counter in enumerate(counters):
        #Tạo mảng một chiều tf
        tf = np.fromiter(counter.values(), dtype=int)

        # Tính w_tf
        w_tf = 1 + np.log10(tf)

        # lấy vị trí của từng từ trong doc theo chỉ mục của bộ từ điển
        word_indices = [vocabs.index(word) for word in counter.keys()]

        #Gán giá trị w_tf vừa tính được vào đúng vị trí trong ma trận
        matrix[i][word_indices] = w_tf

    # Tạo vector biểu diễn idf  
    df = dict.fromkeys(vocabs, 0)

    # Tính df: Đếm số lượng văn bản xuất hiện từ t trong vocabs
    for row in counters:
        for word in row.keys():
            df[word] += 1

    # N: số văn bản trong bộ dữ liệu
    N = len(list_words_docs)

    # Tính idf
    idf = np.asarray(list(map(lambda word: np.log10(N/df[word]), vocabs)))  
    
    # Tạo ma trận tf-idf
    for i in range(matrix.shape[0]):
        # Lấy từng hàng trong ma trận nhân với idf
        matrix[i] *= idf
    
    return  vocabs, idf, matrix

def computeTFIDF_query(query, vocabs, idf):
    # Tính tf
    counter = Counter(query)

    # Tạo ma trận w_tf
    matrix = np.zeros((1, len(vocabs)))  
    
    # tính w_tf
    for word in counter.keys():
        if word in vocabs:
            matrix[0][vocabs.index(word)] = 1 + np.log10(counter[word])
    return matrix * idf

def cosine(vector_query, matrix):

    # Tính giá trị Cosine = (chia mỗi phần tử cho độ dài vector chứa nó) * vector_query
    vector_query = vector_query / np.sqrt(np.sum(np.square(vector_query)))
    
    for i in range(matrix.shape[0]):
       matrix[i] = matrix[i] / np.sqrt(np.sum(np.square(matrix[i]))) * vector_query

    # gộp theo hàng tương ứng với từng văn bản
    cosineSims = np.sum(matrix, axis = 1)
    
    return  cosineSims, np.argsort(-cosineSims)

if __name__ == "__main__":

    # load và lấy nội dung file dữ liệu văn bản doc-text 
    documents = format_text.load_and_format("./doc-text")
    #loại bỏ stop-words
    list_words_docs = remove_stopwords(documents)
       
    # Tính tf.idf
    vocabs, idf, matrix_tfidf = computeTFIDF(list_words_docs)
    
    # load và lấy nội dung file dữ liệu truy vấn query-text
    queries = format_text.load_and_format("./query-text")
    #loại bỏ stop-words
    list_words_queries = remove_stopwords(queries)

    #tạo biến kết quả 
    result = []

    for i in range(len(list_words_queries)):
        # Tính tf.idf của query
        vector_query = computeTFIDF_query(list_words_queries[i], vocabs, idf)

        # Tính cosine
        scores, indices = cosine(np.array(vector_query), np.array(matrix_tfidf))
        
        # format kết quả để lưu ra file
        s = " "
        result.append(i+1)
        for l in indices[:10] + 1:
            s = s + str(l) + " "
        s += "\n/\n"
        result.append(s)

    # Lưu kết quả vào file rlv-ass.txt
    np.savetxt("./rlv-ass.txt", result, fmt='%s')

    