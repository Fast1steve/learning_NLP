import BMM
import FMM









def cut_words(raw_sentence, words_dic):
    bmm_word_list = BMM.cut_words(raw_sentence, words_dic)
    fmm_word_list = FMM.cut_words(raw_sentence, words_dic)
    bmm_word_list_size = len(bmm_word_list)
    fmm_word_list_size = len(fmm_word_list)
    if bmm_word_list_size != fmm_word_list_size:
        if bmm_word_list_size < fmm_word_list_size:
            return bmm_word_list
        else:
            return fmm_word_list
    else:
        FSingle = 0
        BSingle = 0
        isSame = True
        for i in range(len(fmm_word_list)):
            if fmm_word_list[i] not in bmm_word_list:
                isSame = False
            if len(fmm_word_list[i]) == 1:
                FSingle = FSingle + 1
            if len(bmm_word_list[i]) == 1:
                BSingle = BSingle + 1
        if isSame:
            return fmm_word_list
        elif BSingle > FSingle:
            return fmm_word_list
        else:
            return bmm_word_list


























