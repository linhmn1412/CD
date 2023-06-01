# N19DCCN104 - Trần Thị Trúc Ly
# N19DCCN097 - Nguyễn Thị Mỹ Linh
# N19DCCN024 - Nguyễn Bảo Chính

import numpy as np
import format_text

from nltk.corpus import stopwords
stop_words = stopwords.words('english')


class Inverted_Index:
    def __init__(self, path_dictionary, path_documents, skip=1, optimal=False):
        # thiết lập các thông số ban đầu
        self.dictionary = np.load(path_dictionary, allow_pickle=True).item()
        self.documents = format_text.format(path_documents)
        self.skip = skip
        self.optimal = optimal

    def query(self, str_query):
        # tách các thành phần trong câu query
        tokens = str_query.lower().split()
        tokens = [word for word in tokens if word not in stop_words]

        # Thực hiện câu query
        try:
            if self.optimal == True:
                tokens = self.optimize(tokens)
            listDocID = self.dictionary[tokens.pop(0)]
            while len(tokens) != 0:
                if self.skip == 1:
                    listDocID = self.intersect(
                        listDocID,  self.dictionary[tokens.pop(0)])
                else:
                    listDocID = self.intersectWithSkips(
                        listDocID,  self.dictionary[tokens.pop(0)])

            return listDocID
        except KeyError:
            return []

    # giao
    def intersect(self, p1, p2):
        i = 0
        j = 0
        answer = []
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                answer.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                i += 1
            else:
                j += 1

        return answer

    # giá trị sau bước nhảy
    def val_skip(self, p, i):
        return p[i + self.skip]

    # kiểm tra bước nhảy thỏa độ dài 
    def hasSkip(self, p, i):
        return i + self.skip < len(p)

    # Giao với bước nhảy
    def intersectWithSkips(self, p1, p2):
        i = 0
        j = 0
        answer = []
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                answer.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                if self.hasSkip(p1, i) and self.val_skip(p1, i) <= p2[j]:
                    while self.hasSkip(p1, i) and self.val_skip(p1, i) <= p2[j]:
                        i += self.skip
                else:
                    i += 1
            else:
                if self.hasSkip(p2, j) and self.val_skip(p2, j) <= p1[i]:
                    while self.hasSkip(p2, j) and self.val_skip(p2, j) <= p1[i]:
                        j += self.skip
                else:
                    j += 1
        return answer

    # tối ưu câu truy vấn
    def optimize(self, tokens):
        # xếp theo thứ tự tăng dần độ dài mảng
        return sorted(tokens, key=lambda word: len(self.dictionary[word]))


if __name__ == '__main__':
    path_docs = "./query-text"
    queries = format_text.format(path_docs)

    I_index = Inverted_Index("./inverted_index.npy",
                             "./doc-text", skip=3, optimal=True)

    r = []

    for j, query in enumerate(queries, 1):
        listDocID = I_index.query(query)
        s = " "
        r.append(j)
        for l in listDocID:
            s = s + str(l) + " "
        s += "\n/\n"
        r.append(s)

    np.savetxt("./rlv-ass.txt", r, fmt='%s')
