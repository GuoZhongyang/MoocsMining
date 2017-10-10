# coding: utf-8
import jieba
import os
from scipy import interp
from sklearn.metrics import roc_curve, auc
import pandas as pd
import random as rd
import codecs
import numpy as np
from collections import Counter
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from liblinearutil import *


def tokenize(sentence, grams):
    words = sentence.split()
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["_*_".join(words[i:i + gram])]
    return tokens


def build_dict(f, grams):
    dic = Counter()
    for sentence in open(f).xreadlines():
        dic.update(tokenize(sentence, grams))
    return dic


def process_files(file_pos, file_neg, dic, r, outfn, grams):
    output = []
    for beg_line, f in zip(["1", "-1"], [file_pos, file_neg]):
        for l in open(f).xreadlines():
            tokens = tokenize(l, grams)
            indexes = []
            for t in tokens:
                try:
                    indexes += [dic[t]]
                except KeyError:
                    pass
            indexes = list(set(indexes))
            indexes.sort()
            line = [beg_line]
            for i in indexes:
                line += ["%i:%f" % (i + 1, r[i])]
            output += [" ".join(line)]
    output = "\n".join(output)
    f = open(outfn, "w")
    f.writelines(output)
    f.close()


def compute_ratio(poscounts, negcounts, alpha=1):
    alltokens = list(set(poscounts.keys() + negcounts.keys()))
    dic = dict((t, i) for i, t in enumerate(alltokens))
    d = len(dic)
    print "computing r..."
    p, q = np.ones(d) * alpha, np.ones(d) * alpha
    for t in alltokens:
        p[dic[t]] += poscounts[t]
        q[dic[t]] += negcounts[t]
    p /= abs(p).sum()
    q /= abs(q).sum()
    r = np.log(p / q)
    return dic, r


def write_NB(filename, sent, cut=1):
    if cut == 1:
        f = codecs.open(filename, 'w', encoding='utf-8')
    else:
        f = codecs.open(filename, 'w')
    # f = codecs.open(filename, 'w', encoding='utf-8')
    for i, text in enumerate(sent):
        if cut == 1:
            words = jieba.cut(text)
        else:
            words = text
        tags = [i]
        try:
            w = [x for x in words]
        except:
            w = []
        if cut == 1:
            f.write(u' '.join(w))
        else:
            f.write(' '.join(w))
        f.write('\n')
    f.close()


def DataNor(csvName, ifcut=0):
    Data = pd.read_csv(csvName)
    Data_p = Data[(Data.icol(4) == 1.0)]
    Data_n = Data[(Data.icol(4) == 0)]
    sent_p = list(Data_p.icol(3))
    sent_n = list(Data_n.icol(3))
    rd.shuffle(sent_p)
    rd.shuffle(sent_n)
    write_NB("nbsvm_run/p_train", sent_p[:len(sent_p) / 2], cut=ifcut)
    write_NB("nbsvm_run/p_test", sent_p[len(sent_p) / 2:], cut=ifcut)
    write_NB("nbsvm_run/n_train", sent_n[:len(sent_n) / 2], cut=ifcut)
    write_NB("nbsvm_run/n_test", sent_n[len(sent_n) / 2:], cut=ifcut)


def main(out, ngram, liblinear='liblinear-1.96'):
    ngram = [int(i) for i in ngram]
    print "counting..."
    poscounts = build_dict('p_train', ngram)
    negcounts = build_dict('n_train', ngram)

    dic, r = compute_ratio(poscounts, negcounts)
    print "processing files..."
    process_files('p_train', 'n_train', dic, r, "train-nbsvm.txt", ngram)
    process_files('p_test', 'n_test', dic, r, "test-nbsvm.txt", ngram)

    # trainsvm = os.path.join(liblinear, "train")
    # predictsvm = os.path.join(liblinear, "predict")
    # os.system(trainsvm + " -s 0 train-nbsvm.txt model.logreg")
    # os.system(predictsvm + " -b 1 test-nbsvm.txt model.logreg " + out)


def huagetu():
    y_train, x_train = svm_read_problem('nbsvm_run/train-nbsvm.txt')
    y_test, x_test = svm_read_problem('nbsvm_run/test-nbsvm.txt')
    m = train(y_train, x_train, '-c 4')

    p_label, p_acc, p_val = predict(y_test, x_test, m)
    fpr, tpr, thresholds = roc_curve(y_test, p_val)

    f = open("cut0_ng2.txt", 'w')
    f.write(' '.join([str(y) for y in y_test]))
    f.write('\n')
    f.write(' '.join([str(y[0]) for y in p_val]))
    f.close()

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    mean_tpr += interp(mean_fpr, fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
    mean_tpr[0] = 0.0  # 初始处为0
    roc_auc = auc(fpr, tpr)
    # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr  # 在mean_fpr100个点，每个点处插值插值多次取平均
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC NBSVM')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    ngram = '12'  # ‘12’表示unigram+bigram
    out = 'NBSVM-TEST-BIGRAM'
    DataNor("weiboData/NorData-0.csv", ifcut=1)
    os.chdir("nbsvm_run")
    main('NBSVM-TEST-BIGRAM', ngram)
    os.chdir("..")
    huagetu()
