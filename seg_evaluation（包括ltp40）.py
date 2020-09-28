import time
from collections import Counter

import hanlp
import jieba
import pkuseg
import thulac
from ltp import LTP
from LAC import LAC
import prettytable as pt
from pyhanlp import HanLP
from pyltp import Segmentor



def construct_corpus(infile, repeat_time=1):
    line_list = []
    line_list_construct = []
    for line in open(infile, "r"):
        line = line.replace("\u200b", "").strip().lower().replace(" ", "")
        line = line.strip().lower()
        line_list.append(line)
    for i in range(repeat_time):  
        line_list_construct.extend(line_list)
    return line_list_construct


def load_result(file, delimiter='='):
    result = []
    words_per_line = []
    for line in open(file):
        line1 = line.strip().split(delimiter)
        line2 = list(filter(lambda x: x.replace("\u200b", ""), line1))
        line3 = list(filter(lambda x: x.strip() != '', line2))
        result.append(line3)
        words_per_line.append(len(line3))
    return result, words_per_line


def evaluate(algorithm_seg_file, manual_seg_file):
    number_of_rows = 0
    p_sum = []
    r_sum = []

    # 人工分词的词集
    seg_man, words_per_line_man = load_result(manual_seg_file)

    # 算法分词的词集
    seg_alg, words_per_line = load_result(algorithm_seg_file)

    # 求每行文案人手和算法分词的词语交集
    for i in range(len(seg_man)): 
        intersection_man_alg = Counter(seg_man[i]) & Counter(seg_alg[i])
        intersection_sum = sum(intersection_man_alg.values())  
        number_of_rows += 1
        p_sum.append(intersection_sum / len(seg_alg[i]))
        r_sum.append(intersection_sum / len(seg_man[i]))
        # print("第%s行正确的个数：" % number_of_rows, cro_num)

    precision = sum(p_sum) / len(p_sum)
    recall = sum(r_sum) / len(r_sum)
    f1 = 2 * precision * recall / (precision + recall)
    line_aver_length = str(sum(words_per_line) / len(words_per_line))
    precision = round(precision, 3)
    recall = round(recall, 3)
    f1 = round(f1, 3)
    return precision, recall, f1, line_aver_length


def seg_with_baidu_lac(in_file, out_file_path, manual_seg_file):

    # initialization model
    lac = LAC(mode="seg")

    # save seg_result
    corpus = construct_corpus(in_file)
    print(len(corpus))
    print(out_file_path)
    f = open(out_file_path, "w", encoding='utf-8')
    for line in corpus:
        f.write("=".join(lac.run(line)) + "\n")
        f.flush()

    # test accuracy
    p, r, f1, line_aver_length = evaluate(out_file_path, manual_seg_file)

    # 测试qps
    corpus = construct_corpus(in_file, 500)
    start = time.time()
    for line in corpus:
        lac.run(line)
    end = time.time()
    qps = round(len(corpus)/ (end - start), 2)

    return qps, p, r, f1, line_aver_length


def seg_with_jieba(in_file, out_file_path, manual_seg_file):

    # save seg_result
    corpus = construct_corpus(in_file)
    f = open(out_file_path, "w", encoding='utf-8')
    for line_ in corpus:
        f.write("=".join(jieba.cut(line_, cut_all=False)) + "\n")
        f.flush()

    # test accuracy
    p, r, f1, line_aver_length = evaluate(out_file_path, manual_seg_file)

    # test qps
    corpus = construct_corpus(in_file, 500)
    start = time.time()
    for line in corpus:
        "=".join(jieba.cut(line, cut_all=False))
    end = time.time()
    qps = round(len(corpus)/ (end - start), 2)

    return qps, p, r, f1, line_aver_length


def seg_with_thulac(in_file, out_file_path, manual_seg_file):

    # initialization model
    thu1 = thulac.thulac(seg_only=True)

    # save seg_result
    corpus = construct_corpus(in_file)
    f = open(out_file_path, "w", encoding='utf-8')
    for line in corpus:
        result_t = "".join(thu1.cut(line, text=True))
        result_tl = result_t.replace(" ", "=")
        f.write(result_tl + "\n")
        f.flush()

    # test qps
    corpus = construct_corpus(in_file, 500)
    start = time.time()
    for line in corpus:
        "=".join(thu1.cut(line, text=True))
    end = time.time()
    qps = round(len(corpus) /(end- start), 2)

    # test accuracy
    p, r, f1, line_aver_length = evaluate(out_file_path, manual_seg_file)
    return qps, p, r, f1, line_aver_length


def seg_with_ltp(in_file, out_file_path, manual_seg_file):
    # initialization model
    seg = Segmentor()  # 生成对象
    seg.load("./ltp_data_v3.4.0/cws.model")  # 加载分词语料库

    # save seg_result
    corpus = construct_corpus(in_file)
    f = open(out_file_path, "w", encoding='utf-8')
    for line in corpus:
        f.write("=".join(seg.segment(line)) + "\n")
        f.flush()

    # test qps 百度暂时不计算,因为加了延时
    corpus = construct_corpus(in_file, 500)
    start = time.time()
    for line in corpus:
        "=".join(seg.segment(line))
    end = time.time()
    qps = round(len(corpus) /(end- start), 2)

    # test accuracy
    p, r, f1, line_aver_length = evaluate(out_file_path, manual_seg_file)
    return qps, p, r, f1, line_aver_length


def seg_with_ltp40(in_file, out_file_path, manual_seg_file):
    # initialization model
    ltp = LTP()
    line_list = []

    # save seg_result
    corpus = construct_corpus(in_file)
    f = open(out_file_path, "w", encoding='utf-8')
    for line in corpus:
        line_list.append(line)  # 将每句话变成列表["Xxxx"]
        seg_result, hidden = ltp.seg(line_list)
        f.write("=".join(seg_result[0]) + "\n")
        line_list.clear()
        f.flush()

    # test qps
    corpus = construct_corpus(in_file, 1)
    start = time.time()
    for line in corpus:
       segment, hidden = ltp.seg(list(line))
    end = time.time()
    qps = round(len(corpus) /(end- start), 2)

    # test accuracy
    p, r, f1, line_aver_length = evaluate(out_file_path, manual_seg_file)
    return qps, p, r, f1, line_aver_length



def seg_with_han176(in_file, out_file_path, manual_seg_file):
    
    # save seg_result
    corpus = construct_corpus(in_file)
    f = open(out_file_path, "w", encoding='utf-8')
    for line in corpus:
        result_h176 = "=".join("%s" % t.word for t in HanLP.segment(line))  # 每个text是一句话
        f.write(result_h176 + "\n")
        f.flush()

    # test qps
    corpus = construct_corpus(in_file, 500)
    start = time.time()
    for line in corpus:
        _ = HanLP.segment(line)
    end = time.time()
    qps = round(len(corpus) /(end - start), 2)

    # test accuracy
    p, r, f1, line_aver_length = evaluate(out_file_path, manual_seg_file)
    return qps, p, r, f1, line_aver_length


def seg_with_han200(in_file, out_file_path, manual_seg_file):
    # initialization model
    tokenizer = hanlp.load("PKU_NAME_MERGED_SIX_MONTHS_CONVSEG")  # 以默认配置加载模型

    # save seg_result
    corpus = construct_corpus(in_file)
    f = open(out_file_path, "w", encoding='utf-8')
    for line in corpus:
        f.write("=".join(tokenizer(line)) + "\n")
        f.flush()

    # test qps 百度暂时不计算,因为加了延时
    corpus = construct_corpus(in_file, 1)
    start = time.time()
    for line in corpus:
        tokenizer(line)
    end = time.time()
    qps = round(len(corpus) /(end- start), 2)

    # test accuracy
    p, r, f1, line_aver_length = evaluate(out_file_path, manual_seg_file)
    return qps, p, r, f1, line_aver_length


def seg_with_pku(in_file, out_file_path, manual_seg_file):
    # initialization model
    seg = pkuseg.pkuseg()  # 以默认配置加载模型

    # save seg_result
    corpus = construct_corpus(in_file)
    f = open(out_file_path, "w", encoding='utf-8')
    for line in corpus:
        f.write("=".join(seg.cut(line)) + "\n")
        f.flush()

    # test qps
    corpus = construct_corpus(in_file, 500)
    start = time.time()
    for line in corpus:
        seg.cut(line)
    end = time.time()
    qps = round(len(corpus) /(end- start), 2)
    print("pku:", qps)

    # test accuracy
    p, r, f1, line_aver_length = evaluate(out_file_path, manual_seg_file)
    return qps, p, r, f1, line_aver_length


if __name__ == '__main__':

    tb = pt.PrettyTable()
    tb.field_names = ['分词工具', 'QPS', '精度', '召回', 'F1', '句子平均词数']
    test_list = [
        # (seg_with_baidu_lac, '_baidu_lac.txt', '百度LAC'),
        (seg_with_jieba, '_jieba.txt', '结巴分词'),
        (seg_with_thulac, '_thulac.txt', '清华thulac'),
        (seg_with_ltp, '_ltp.txt', '哈工大ltp'),
        (seg_with_ltp40, '_ltp40.txt', '哈工大ltp40'),
        (seg_with_pku, '_pku.txt', '北大pku'),
        (seg_with_han176, '_hanlp_v1.txt', 'HanLP_v1'),
        (seg_with_han200, '_hanlp_v2.txt', 'HanLP_v2')
    ]

    output_file = "./seg_result/corpus_"
    file_list = [('./original_corpus/corpus_ag.txt', './manual/corpus_ag_manual.txt'),
                 ('./original_corpus/corpus_weibo.txt', './manual/corpus_weibo_manual.txt'),
                 ('./original_corpus/corpus_video.txt', './manual/corpus_video_manual.txt')]
    for input_file, manual_file in file_list:
        lst = ['///////' for n in range(6)]
        lst[0] = "语料为："
        lst[1] = input_file.split(".")[-2].split("_")[-1]
        tb.add_row(lst)
        for func, suffix, name in test_list:
            result = func(input_file, output_file + input_file.split(".")[-2].split("_")[-1] + suffix, manual_file)
            tb.add_row([name] + list(result))  # 合并列表

        print(tb)
