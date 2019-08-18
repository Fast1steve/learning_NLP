

def getWordDic():
    """
    读取词典文件
    载入词典
    :return:
    """
    words_dic = []
    with open("./dict.txt", "r",) as dic_input:
        for word in dic_input:
            words_dic.append(word.strip())

    return words_dic