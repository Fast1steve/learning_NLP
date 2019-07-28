import nltk
import string
from nltk.corpus import stopwords
from collections import Counter
from gensim import corpora, models, matutils
import math
import numpy as np


text1 = """
Football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal. 
Unqualified, the word football is understood to refer to whichever form of football is the most popular 
in the regional context in which the word appears. Sports commonly called football in certain places 
include association football (known as soccer in some countries); gridiron football (specifically American 
football or Canadian football); Australian rules football; rugby football (either rugby league or rugby union); 
and Gaelic football. These different variations of football are known as football codes.
"""

text2 = """
Basketball is a team sport in which two teams of five players, opposing one another on a rectangular court, 
compete with the primary objective of shooting a basketball (approximately 9.4 inches (24 cm) in diameter) 
through the defender's hoop (a basket 18 inches (46 cm) in diameter mounted 10 feet (3.048 m) high to a backboard 
at each end of the court) while preventing the opposing team from shooting through their own hoop. A field goal is 
worth two points, unless made from behind the three-point line, when it is worth three. After a foul, timed play stops 
and the player fouled or designated to shoot a technical foul is given one or more one-point free throws. The team with 
the most points at the end of the game wins, but if regulation play expires with the score tied, an additional period 
of play (overtime) is mandated.
"""

text3 = """
Volleyball, game played by two teams, usually of six players on a side, in which the players use their hands to bat a 
ball back and forth over a high net, trying to make the ball touch the court within the opponents’ playing area before 
it can be returned. To prevent this a player on the opposing team bats the ball up and toward a teammate before it touches 
the court surface—that teammate may then volley it back across the net or bat it to a third teammate who volleys it across 
the net. A team is allowed only three touches of the ball before it must be returned over the net.
"""

# 文本预处理
# 函数：text文件分句，分词，并去掉标点


def get_tokens(text):
    text = text.replace('\n', '')
    sents = nltk.sent_tokenize(text)   #分句
    tokens = []
    for sent in sents:
        for word in nltk.word_tokenize(sent):
            if word not in string.punctuation:
                tokens.append(word)
    return tokens


# 对原始的text文件去掉停用词
# 生成count字典，即每个单词的出现次数
def make_count(text):
    tokens = get_tokens(text)
    filtered = [w for w in tokens if not w in stopwords.words('english')] #去掉停用词
    count = Counter(filtered)
    return count


# training by gensim's Ifidf Model
def get_words(text):
    tokens = get_tokens(text)
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    return filtered

# get text


count1, count2, count3 = get_words(text1), get_tokens(text2), get_words(text3)
countlist = [count1, count2, count3]
# training by TfidfModel in gensim
dictionary = corpora.Dictionary(countlist)
new_dict = {v: k for k, v in dictionary.token2id.items()}
corpus2 = [dictionary.doc2bow(count) for count in countlist]
tfidf2 = models.TfidfModel(corpus2)
corpus_tfidf = tfidf2[corpus2]

# output


print("\nTraining by gensim Tfidf Model.......\n")
for i, doc in enumerate(corpus_tfidf):
    print("Top words in document %d" % (i + 1))
    sorted_words = sorted(doc, key=lambda x: x[1], reverse=True)  # type=list
    for num, score in sorted_words[:3]:
        print("    Word: %s, TF-IDF: %s" % (new_dict[num], round(score, 5)))


# 计算tf
def tf(word, count):
    return count[word]/sum(count.values())


# 计算count_list有多少个文件包含word
def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)


# 计算idf
def idf(word, count_list):
    return math.log2(len(count_list) / (n_containing(word, count_list)))    #对数以2为底


# 计算tf-idf
def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)


# 对向量做规范化, normalize
def unitvec(sorted_words):
    lst = [item[1] for item in sorted_words]
    L2Norm = math.sqrt(sum(np.array(lst)*np.array(lst)))
    unit_vector = [(item[0], item[1]/L2Norm) for item in sorted_words]
    return unit_vector


# TF-IDF 测试
count1_2 = make_count(text1)
count2_2 = make_count(text2)
count3_2 = make_count(text3)
countlist_2 = [count1_2, count2_2, count3_2]
print("Training by original algorithm......\n")
for i, count in enumerate(countlist_2):
    print("Top words in document %d"%(i+1))
    scores = {word: tfidf(word, count, countlist_2) for word in count}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)  # type=list
    sorted_words = unitvec(sorted_words)
    for word, score in sorted_words[:3]:
        print("    Word: %s, TF-IDF: %s" % (word, round(score, 5)))






