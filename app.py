from flask import Flask, render_template, request
import utlis
import FMM
import jieba

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/getResult', methods=['POST'])
def getResult():
    source = request.form.get("source")
    words_dic = utlis.getWordDic()
    words_list1 = FMM.cut_words(source, words_dic)
    words_list = jieba.cut(source)
    return "/".join(words_list)
if __name__ == '__main__':
    app.run()
