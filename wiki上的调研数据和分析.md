中文分词工具调研
 E编辑 F收藏 W关注中 S分享
页面… NLP
跳到banner的尾部
回到标题开始
转至元数据结尾
由 李百川创建, 最终由 黄于晏修改于 2020-07-29转至元数据起始
中文分词工具
分词是预处理句子和篇章时重要的一个步骤，由于中文不像英文每个单词之间都有空格，所以我们需要专门的工具对中文进行分词。

在实际任务中，我们发现最常碰到的场景是对新媒体短文本进行分词，如微博的博文、短视频标题、广告文案，以下对几个常见的分词工具做了汇总，并手动标注了一些短文本语料测试其效果。

工具简介
一、百度中文开源词法分析工具(LAC) （https://github.com/baidu/lac）
1、介绍：百度自然语言处理部研发的LA一个联合的词法分析模型，整体性地完成中文分词、词性标注、专名识别任务。适用场景更多是与实体识别相关的，比如知识图谱，知识问答，信息抽取等，也可以作为其他模型算法的基础工具。（2.0版本具体介绍：https://mp.weixin.qq.com/s/ePYwprZd4NbvGkdtOgrI7w）

2、算法：LAC基于一个堆叠的双向GRU结构,在长文本上准确复刻了百度AI开放平台上的词法分析算法。

3、使用须知：

（1）环境：代码对 Python 2/3 均兼容

（2）版本：jieba - 2.0.4

（3）安装：pip install LAC

之前需要通过PaddleNLP或者PaddleHub调用lac，现在可直接调用，相当方便
（4）使用：

from LAC import LAC
lac = LAC(mode='seg')  # 装载分词模型【lac = LAC(mode='lac')  # 装载LAC模型】
text = u"LAC是个优秀的分词工具"  # 输入为Unicode编码的字符串
texts = [u"分词工具", u"百度是一家公司"]  #  输入为多个句子组成的list，平均速率更快
lac_result = lac.run(texts)

二、结巴分词（https://github.com/fxsjy/jieba）
1、介绍：“结巴”分词是一个Python 中文分词组件，是Github上点赞最多的(Python)中文分词工具，优点是速度非常快。主要通过词典来进行分词及词性标注，两者使用了一个相同的词典。分词的结果优劣将很大程度上取决于词典，虽然使用了HMM来进行新词发现。

2、算法：（jieba分词详解：https://www.jianshu.com/p/2cccb07d9a4e）

基于前缀词典实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图 (DAG)
采用了动态规划查找最大概率路径, 找出基于词频的最大切分组合
对于未登录词，采用了基于汉字成词能力的 HMM 模型，使用了 Viterbi 算法
3、使用须知：

（1）环境：代码对 Python 2/3 均兼容

（2）版本：jieba - 0.42.1

（3）安装：pip install jieba

如果需要使用paddle模式下的分词和词性标注功能，请先安装paddlepaddle-tiny
pip install paddlepaddle-tiny==1.6.1。
PaddlePaddle深度学习框架，训练序列标注（双向GRU）网络模型实现分词
（4）使用：

import jieba
#  jieba.enable_paddle()  可选启动paddle模式
jieba.cut("我来到北京清华大学", cut_all=False)    # 精确模式
缺点

        延迟加载，每次分词要从文件中读取模型，



三、清华THULAC（http://thulac.thunlp.org/）
1、介绍：清华大学自然语言处理与社会人文计算实验室研制推出的一套中文词法分析工具包，具有中文分词和词性标注功能，速度也很快。

分词粒度只有一种，简单直接。

2、算法：结构化感知器(Structured Perceptron, SP)，链接：http://www.cnblogs.com/en-heng/p/6429355.html

3、使用须知：

（1）接受编码：仅处理UTF8编码中文文本

（2）版本：thulac - 0.2.1

（3）安装：pip install thulac

（4）使用：

import thulac
thu1 = thulac.thulac() #默认模式
text = thu1.cut("我爱北京天安门", text=True) #进行一句话分词
print(text)
四、哈工大LTP（https://github.com/HIT-SCIR/ltp-cws）
介绍：LTP是哈工大开源的一套中文NLP工具，涵盖了分词、词性标注、命名实体识别、依存句法分析、语义角色标注、语义依存分析等功能。

分词粒度也只有一种，简单直接。

（一）LTP - 3.4.0（https://github.com/HIT-SCIR/pyltp）
1、算法：结构化感知器，链接：https://www.cnblogs.com/en-heng/p/9167873.html。

2、使用须知：

（1）运行环境：

Py<3.4：windowsVS2015和VS2017不支持
Py>3.4：windowsVS2015和windowsVS2017，Linux，mac都支持
（2）版本对应：

pyltp 版本：0.2.0
LTP 版本：3.4.0
模型版本：3.4.0
（3）安装：pip install pyltp

（4）使用：（https://pyltp.readthedocs.io/zh_CN/latest/）

from pyltp import Segmentor
segmentor = Segmentor()
segmentor.load("/path/to/your/cws/model")
words = segmentor.segment("元芳你怎么看")
（二）LTP - 4.0（https://github.com/HIT-SCIR/ltp#RELTRANS）
1、算法：分词: Electra Small1 + Linear

LTP 4.0 (Base(v2)) 
LTP 4.0 (Small(v2))
LTP 4.0 (Tiny(v2))
2、使用须知：

（1）运行环境：

Python 3.7
LTP 4.0 Batch Size = 1
（2）版本：ltp-4.0.6.post1

（3）安装：pip install ltp

（4）使用：

from ltp import LTP
ltp = LTP()      # 默认加载 Small 模型，没有的会直接下载
segment, hidden = ltp.seg(["他叫汤姆去拿外衣。"])    #切分的text必须是list
print(segment)      # 输出[['他', '叫', '汤姆', '去', '拿', '外衣', '。']] 
（5）论坛：http://ltp.ai/faq.html

五、汉语言处理包HanLP
介绍：HanLP是由一系列模型预算法组成的工具包，结合深度神经网络的分布式自然语言处理，具有功能完善、性能高效、架构清晰、语料时新、可自定义等特点，提供词法分析、句法分析、文本分析和情感分析等功能，

（一）1.x版本（https://github.com/hankcs/HanLP/tree/1.x)
1、介绍：pkuseg是由北京大学语言计算与机器学习研究组研制推出的一套全新的中文分词工具包

2、算法：

（1）HMM-Bigram（速度与精度最佳平衡；一百兆内存）：

最短路分词 （viterbi算法https://www.cnblogs.com/hapjin/p/11172299.html）
N-最短路分词
（2）由字构词（侧重精度，全世界最大语料库，可识别新词；适合NLP任务）：

感知机分词
CRF分词
（3）词典分词（侧重速度，每秒数千万字符；省内存）：

极速词典分词
3、使用须知：

（1）运行环境：要求Python 3.6以上，支持Windows，可以在CPU上运行，推荐GPU/TPU。

（2）版本：pyhanltp-0.1.66 【对应hanlp-1.7.8】

（3）安装：pip install pyhanltp

（4）使用：

from pyhanlp import HanLP
HanLP.segment("你好，欢迎使用HanLP汉语处理包"） #HMM-Bigram

（二）2.X版本（https://github.com/hankcs/HanLP）
1、介绍：HanLP 2.0正处于alpha测试阶段，未来将实现知识图谱、问答系统、自动摘要、文本语义相似度、指代消解、三元组抽取、实体链接等功能。

2、算法：基于深度学习模型

分词模型：PKU_NAME_MERGED_SIX_MONTHS_CONVSEG （https://file.hankcs.com/hanlp/cws/pku98_6m_conv_ngram_20200110_134736.zip）
3、使用须知：

（1）运行环境：, 基于tensorflow2.0。

（2）版本：hanlp-2.0.0a46 

（3）安装：pip install hanlp

（4）使用：

import hanlp
tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
seg_result = tokenizer(text)
（5）论坛：http://bbs.hankcs.com



六、北大pkuseg(https://github.com/lancopku/PKUSeg-python)
1、介绍：pkuseg是由北京大学语言计算与机器学习研究组研制推出的一套全新的中文分词工具包

2、特点：支持细分领域分词，为不同领域的数据提供个性化的预训练模型，支持用户使用全新的标注数据进行训练，运行速度中等偏下。

提供的预训练模型：新闻领域，网络（微博）领域，医药领域，旅游领域，以及混合领域。
3、使用须知：

（1）运行环境：仅支持linux(ubuntu)、mac、windows 64 位的python3环境

（2）版本：pkuseg-0.0.25

（3）安装：pip install pkuseg

更新到新版本：pip3 install -U pkuseg【镜像源更换到阿里源】
（4）使用：如果用户无法确定分词领域，推荐使用默认模型分词

import pkuseg
seg = pkuseg.pkuseg()  text = seg.cut('z')
若要加载特定领域模型  pkuseg.pkuseg(model_name='medicine')                

分词工具性能比较                                                                                                                                                                                                              
工具
多进程分词
分词语料格式
功能
自训练模型
用户自定义词典
优点
缺点
百度LAC	不支持	
所有数据文件均为UT-8编码

分词，词性标注与实体识别

支持

提供了增量训练的接口，数据文件UTF-8编码，词法分析和分词都可训练。分词数据，使用空格作为单词切分标记。词法分析在分词数据的基础上，每个单词以“/type”的形式标记其词性或实体类别

支持

装载干预词典：定制专名类型（个性化标签）和切分结果，例：春天/SEASON。

词典格式：每行表示一个定制化的item，例：花/n 开/v



分词粒度是实体级别，有实体识别的效果。适用于知识图谱、问答、信息抽取。

1.0版本安装麻烦，依赖paddle库。（2.0开始为独立的python模块，使用方便）

分词粒度较大，如要面向搜索的分词，需要用户微调模型。

结巴分词	
支持

并行分词仅支持默认分词器 和默认词性标注分词器

例子

unicode 、UTF8编码字符串

分词模式：

精确分词
全模式（所有可能词语）
搜索引擎模式（长句切分）
新词识别，繁体分词，

不支持

支持

词典格式：[词语] [词频(可省)] [词性(可省)]，用空格隔开，UTF8 编码

使用简单，速度快

部分词语成词概率过高或过低，可以强制调节词频解决。



延迟加载机制import jieba 和 jieba.Tokenizer() 不会立即触发词典的加载，手动初始化：jieba.initialize() ，延迟加载机制后，你可以改变主词典的路径。



清华thulac	不支持	UTF8编码中文文本	
一种粒度分词。

分词可选是否有词性标注的模式，是否过滤无意义词，繁体转简体。

支持

提供模型训练程序train_c，用斜线分割或者已进行词性标注的训练集可以训练分词和词性标注模型。新模型覆盖旧模型即可使用。

支持

用户词典中的词会被打上uw标签。词典中每一个词一行，UTF8编码

直接对文件进行切分，输出分词后的文件，操作简单。	速度中下，准确率不高
哈工大ltp	不支持	所有输入的分析文本和输出的结果的编码均为 UTF8。	词性标注、命名实体识别、依存句法分析、语义角色标注、语义依存分析	
支持

支持使用用户训练好的个性化模型，训练需使用 LTP。

使用时要同时添加分词模型和用户的增量模型。也可同时加入外部词典

支持

分词、词性标注支持用户使用自定义词典。

分词外部词典：文本文件，每行一词，编码 UTF-8

词性标注外部词典：文本文件，每行一词，第一列指定单词，第二列指定该词的候选词性（可多项，一项占一列），列间用空格区分。



速度和准确率中上

提供个性化分词模型，在原有数据基础上进行增量训练。 利用领域丰富数据，又兼顾目标领域特殊性。

外部词典以特征方式加入机器学习算法，不是词典匹配，无法直接增加某词的词频，不能保证词按照词典的方式进行切分。

哈工大ltp40	
不支持

utf8
北大pku	
支持

多进程分词修改

utf8	
混合领域分词，细分领域分词，词性标注

支持

预训练模型

支持

在使用默认词典的同时会额外使用用户自定义词典。

词典格式：一行一词（如果选择进行词性标注并且已知该词的词性，则在该行写下词和词性，中间用tab字符隔开）

速度和准确率是其他分词工具属于中等。

提供个性化的多领域预训练模型切换，适合专业领域应用。精细化场景

运行速度一般，不过官方正在优化中，通用模型的分词精度没有jieba好。
HanLP_v1	
支持

并行切分多个句子，单需消耗更多GPU显存。

待切分的字符串要用 list转换为字符列表	
分词模式：

HMM-Bigram（速度与精度平衡；一百兆内存）
由字构词（侧重精度，可识别新词；适合NLP任务）
词典分词（侧重速度，每秒数千万字符；省内存）
词性标注，命名实体识别，自动摘要，短语提取，关键词提取，简繁转换，文本推荐，依存句法分析，文本的分类和聚类，word2vec工具



支持

部分默认模型训练自小型语料库，鼓励用户自行训练。所有模块提供训练接口，语料可参考98年人民日报语料库。

支持

核心词性词频词典：可以随时增删，影响全部分词器。

格式：每行一词[单词] [词性A] [A的频次]，词性不填默认为n

核心二元文法词典：依赖上一个词典，储存两个词的接续

速度和准确率是其他分词工具中最高。

在提供丰富功能的同时，HanLP内部模块坚持低耦合、模型坚持惰性加载、服务坚持静态提供、词典坚持明文发布，使用非常方便。默认模型训练自全世界最大规模的中文语料库，同时自带一些语料处理工具，帮助用户训练自己的模型。

词典文件添加自定义词典没有结巴快。hanlp支持自定义词典中带有空格的词，jieba不支持。
HanLP_v2	unknown	utf8	
任意语种分词和词性标注、命名实体识别、依存句法分析、语义依存分析

支持

训练脚本，训练案例

支持

参考demo挂载用户词典

准确率中上HanLP2.0提供许多预训练模型，而终端用户仅需两行代码即可部署，面向多语种。未来将实现知识图谱、问答系统、自动摘要、文本语义相似度、指代消解、三元组抽取、实体链接等功能。	模型加载过慢，测试中，性能还没有达到生产效果


测试设置
环境
Linux version 4.15.0-29-generic

操作系统：Ubuntu 18.04.1 LTS

处理器：Intel® Core™ i3-4150 CPU @ 3.50GHz × 4

内存：7.7 GiB

Python 3.6.10

各工具版本和设置
工具	版本	设置	备注
百度模型(LAC)	2.0.4	无额外用户词典	使用默认模型，lac = LAC(mode='seg')，seg_result = lac.run(text)
结巴	0.42.1	无额外用户词典	使用默认模型，jieba.lcut(text, cut_all=False)
清华THULAC	0.2.1	无额外用户词典	使用默认模型，thu.cut(text, text=True)
哈工大pyLTP	v3.4.0	无额外用户词典	
使用模型：ltp_data_v3.4.0/cws.model，ltp.segment(text)

4.0.6	
使用模型：Electra Small1，segment, hidden = ltp.seg(["text"])，text需为LIst格式

HanLP	
1.7.8



无额外用户词典	HanLP.segment()
2.0.0a46

使用预训练模型：tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')

tokenizer(text)

北大pkuseg	0.0.25	无额外用户词典	使用默认模型及默认词典pkuseg.pkuseg()，默认情况下使用的预模型是msra


由于几百条数据对于分词工具来说实在太少，为了测试分词速度，我们将语料重复了100遍，最后计算各工具分词的QPS（不包括加载模型）。

语料
广告文案200条

微博100条

短视频标题100条

测试代码
seg_evaluation.py

测试结果
广告文案	
corpus_ag.txt

corpus_ag_manual.txt

corpus_ag_baidu_lac.txt

corpus_ag_jieba.txt	corpus_ag_thulac.txt	corpus_ag_ltp.txt	
corpus_ag_ltp40.txt

corpus_ag_hanlp_v1.txt	corpus_ag_hanlp_v2.txt	corpus_ag_pku.txt
微博	
corpus_weibo.txt

corpus_weibo_manual.txt

corpus_weibo_baidu_lac.txt

corpus_weibo_jieba.txt

corpus_weibo_thulac.txt

corpus_weibo_ltp.txt

corpus_weibo_ltp40.txt	corpus_weibo_hanlp_v1.txt	corpus_weibo_hanlp_v2.txt	corpus_weibo_pku.txt
短视频标题	
corpus_video.txt

corpus_video_manual.txt

corpus_video_baidu_lac.txt

corpus_video_jieba.txt	corpus_video_thulac.txt	
corpus_video_ltp.txt

corpus_video_ltp40.txt	corpus_video_ltp.txt	corpus_video_hanlp_v2.txt	corpus_video_pku.txt

广告文案
人工分词200条，平均每条分词12.60个

百度LAC	620.42	0.771	0.818	0.794	13.625	
结巴分词	5447.29	0.745	0.817	0.779	14.02	
清华thulac	1023.57	0.634	0.744	0.685	15.255	
哈工大ltp	3087.81	0.699	0.797	0.745	14.75	
哈工大ltp40	21.12	0.689	0.793	0.737	14.91	
北大pku	1014.45	0.73	0.815	0.77	14.335	
HanLP_v1	6584.11	0.722	0.812	0.764	14.49	
HanLP_v2	5.22	0.699	0.788	0.741	14.535	
微博
人工分词100条，平均每条分词21.59个

百度LAC	485.95	0.684	0.727	0.705	23.77	
结巴分词	2536.49	0.661	0.731	0.694	24.8	
清华thulac	672.41	0.536	0.648	0.587	26.49	
哈工大ltp

2448.51	0.623	0.702	0.66	25.11	
哈工大ltp40	26.99	0.621	0.712	0.663	25.73	
北大pku	704.77	0.648	0.718	0.681	24.64	
HanLP_v1	6879.74	0.653	0.744	0.696	25.37	
HanLP_v2	5.15	0.618	0.698	0.656	25.04	
短视频
人工分词100条，平均每条分词10.03个

百度LAC	1157.55	0.671	0.736	0.702	11.37	
结巴分词	8381.82	0.648	0.734	0.689	11.81	
清华thulac	1475.98	0.555	0.668	0.606	12.77	
哈工大ltp	5106.72	0.632	0.72	0.673	12.0	
哈工大ltp40	21.14	0.607	0.714	0.656	12.4	
北大pku	1477.2	0.647	0.731	0.686	11.71	
HanLP_v1	10927.29	0.665	0.757	0.708	11.82	
HanLP_v2	5.76	0.62	0.724	0.668	12.34	
结论
按照平均分词效果来看，百度模型效果最好，HanLP_v1其次，结巴再之，哈工大LTP第四。ltp4和hanlp2.x正在测试优化中，期待进一步的效果。

综合速度和准确率，以及其他词法分析的功能完善程度来看，HanLP_v1无疑是首选。
