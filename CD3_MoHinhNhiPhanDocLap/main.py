# N19DCCN104 - Trần Thị Trúc Ly
# N19DCCN097 - Nguyễn Thị Mỹ Linh
# N19DCCN024 - Nguyễn Bảo Chính

import numpy as np
from collections import Counter
from nltk.corpus import stopwords
import format_text
import pandas as pd

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
def marker_matrix(documents ,vocabs):
    # tạo ma trận đánh dấu
    matrix = np.zeros((len(documents), len(vocabs)))

    for i in range(len(documents)):
        for word in documents[i].split():
            if word not in stop_words:
                matrix[i][vocabs.index(word)] = 1
    return matrix
class BinIndependenceModel:
    def __init__(self, documents, matrix, vocabs):
        self.documents = documents
        self.matrix = matrix
        self.vocabs = vocabs
        
        # Khởi tạo các tham số ban đầu
        # N: số lượng văn bản trong kho dữ liệu
        self.N = len(documents)

        # n : biểu diễn số lượng văn bản chứa xi trong vocabs
        self.n = np.sum(matrix, axis = 0)

        # Trong trường hợp này ct tương tự trọng số idf
        self.ct = np.log((self.N - self.n + 0.5) / (self.n + 0.5))

        self.N_rel = 10 # Top xếp hạng
        self.n_loop = 10 # số vòng lặp tối đa
        self.threshold = 0.00001 # ngưỡng cho độ chênh lệch giữa d(c_new,c_old) < threshold
    
    # Xếp hạng
    def ranking(self, vectorQuery):
        indices = np.where(vectorQuery == 1) 
        scores = np.sum(self.matrix[:, *indices] * self.ct[indices], axis = 1)

        # trả về scores của các docs theo thứ tự giảm dần    
        ranks = np.argsort(-scores)

        return ranks, scores[ranks]

    def recompute_ct(self, relevant_doc, vectorQuery):
        # index of qi = 1
        qi_1 = np.where(vectorQuery == 1) 
        # Số văn bản có chứa xi trong V
        n_vi = np.sum(self.matrix[relevant_doc][:,*qi_1], axis = 0) 
        
        pi = (n_vi + 0.5) /( self.N_rel + 1) 

        n = self.n[*qi_1] # df

        ri = (n - n_vi + 0.5) / (self.N - self.N_rel + 1) 

        # cập nhật lại ct
        self.ct[qi_1] = np.log(pi/(1-pi)) - np.log(ri/(1-ri)) 

    def handle_query(self, vectorQuery):
        # Thực hiện lặp đến khi nào hội tụ 
        # Tức đến khi giá trị của pi và ri không thay đổi nhiều (d(c_new,c_old) < threshold)  
        # hoặc đã lặp đến số lần tối đa n_loop
        for i in range(self.n_loop):
                ranks, _ = self.ranking(vectorQuery)
                ct_old = np.array(self.ct)
                self.recompute_ct(ranks[:self.N_rel], vectorQuery) # VR = V
                if i == self.n_loop - 1 or np.sqrt(np.sum((self.ct - ct_old)**2, axis = 0)) < self.threshold:
                    return self.ranking(vectorQuery)
if __name__ == "__main__":
    # load và lấy nội dung file dữ liệu văn bản doc-text 
    documents = format_text.load_and_format("./doc-text")
    
    #loại bỏ stop-words
    list_words_docs = remove_stopwords(documents)

    # Tạo tập từ điển
    vocabs = list(set.union(*map(set, list_words_docs)))

    # Tạo ma trận đánh dấu
    matrix = marker_matrix( documents ,vocabs)

    # load và lấy nội dung file dữ liệu truy vấn query-text
    queries = format_text.load_and_format("./query-text")
    #loại bỏ stop-words
    list_words_queries = remove_stopwords(queries)

    #tạo biến kết quả 
    result = []

    bim = BinIndependenceModel(documents, matrix, vocabs)
    with open('./rlv-ass.txt' ,'w') as f:
        for i, query in enumerate(list_words_queries):
            f.write(f"Query {i+1} \n")        
            # Tạo vector đánh dấu cho query
            vectorQuery = np.isin(vocabs, query)

            # Tính score, xếp hạng doc
            ranking, scores = bim.handle_query(vectorQuery)

            #format kết quả lưu vào file
            for index, score in list(zip(ranking, scores))[:bim.N_rel]:
                f.write(f"\tDoc {index+1} - Score: {score} \n")
            f.write("\t/\n")


   

    

    





    