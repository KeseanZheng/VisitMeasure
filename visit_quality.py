import pandas as pd
import numpy as np
import jieba
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Word2Vec
from scipy.spatial import distance
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
import os
from scipy.stats.mstats import winsorize
import warnings
warnings.filterwarnings('ignore')

pd.set_option('expand_frame_repr', False) #不允许换行显示
pd.set_option('display.max_rows', 500)

def data_clear():
    # 读取调研Q&A明细表
    df1 = pd.read_excel("D:\data\投资者调研\调研提问&解答表220148796(仅供混沌书苑使用)\IRM_MEETINGMINUTES.xlsx")
    df2 = pd.read_excel("D:\data\投资者调研\调研提问&解答表220148796(仅供混沌书苑使用)\IRM_MEETINGMINUTES1.xlsx")
    df = pd.concat([df1, df2], axis=0)
    df = df[(df['ReportDate'] >= '2013-01-01') & (df['ReportDate'] <= '2022-12-31')]
    df['ReportYear'] = pd.to_datetime(df['ReportDate']).dt.year
    df['ReportMonth'] = pd.to_datetime(df['ReportDate']).dt.month
    df['Q&A'] = df['Question'] + df['Answer']

    # 读取实地调研数据
    basic_info = pd.read_excel("D:\data\投资者调研\调研纪要文本表225833289(仅供混沌书苑使用)\IRM_ACTIVERECORD.xlsx", usecols=['ReportID', 'Address'])
    keywords = ['线上', '网上', '程序', '在线', '网络', '直播', '网站', '电话', 'http', 'www', '平台', '微信', '视频', '邮件']
    def WhetherOnline(addr):
        if pd.notnull(addr) and any(keyword in addr for keyword in keywords):
            return 1
        else:
            return 0
    basic_info['Online'] = basic_info['Address'].apply(WhetherOnline)
    df = df.merge(basic_info, on=['ReportID'], how = 'left')
    df = df[df['Online'] == 0]

    df.to_csv('data\Q&A.csv', index=False, encoding='utf-8-sig')

def text_cut():
    # 读取数据
    df = pd.read_csv('data\Q&A.csv')
    df2 = df.groupby(['Symbol','ReportYear'])['ReportID'].count().reset_index()
    print(df2)
    exit()
    # print(len(set(df['ReportID'])))
    # print(len(set(df['Symbol'])))
    # print(len(df)/len(set(df['ReportID'])))
    # exit()
    def remove_non_chinese(string):
        if not isinstance(string, str):
            return ''
        return re.sub(r'[^\u4e00-\u9fa5]', '', string)
    # 对Views列应用remove_non_chinese函数，删除除汉字以外的字符
    df['Q&A_clear'] = df['Q&A'].apply(remove_non_chinese)
    df['Question_clear'] = df['Question'].apply(remove_non_chinese)
    df['Answer_clear'] = df['Answer'].apply(remove_non_chinese)
    # 加载停用词列表
    with open('..\stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    # 对Question列进行分词并去停用词
    jieba.load_userdict(r"..\terms\Finwords.txt")
    df['Question_cut'] = df['Question_clear'].apply(lambda x: " ".join([word for word in jieba.cut(x) if word not in stopwords]))
    df['Answer_cut'] = df['Answer_clear'].apply(lambda x: " ".join([word for word in jieba.cut(x) if word not in stopwords]))
    df['Q&A_cut'] = df['Q&A_clear'].apply(lambda x: " ".join([word for word in jieba.cut(x) if word not in stopwords]))

    df.dropna(inplace=True)

    df.to_csv('data\Q&A_cut.csv', index=False, encoding='utf-8-sig')

def doc2vec():
    df = pd.read_csv(r'data\Q&A_cut.csv')
    # 设置“ReportID_Rank”为索引
    df['ReportID_Rank'] = df['ReportID'].astype(str) + '_' + df['Rank'].astype(str)
    df.set_index('ReportID_Rank', inplace=True)

    # 使用gensim的doc2vec库将分词后的文本转为向量
    for text_type in ['Question','Answer','Q&A']:
        documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(df[text_type+'_cut'])]
        model = Doc2Vec(documents, vector_size=300, window=2, min_count=1, workers=4)
        model.save('model\doc2vec_' + text_type)
        print("模型训练完成")
        model = Doc2Vec.load('model\doc2vec_' + text_type)
        # 创建一个词典，词典的key为索引值，Value为向量
        vector_dict = {index: model.infer_vector(doc.split()) for index, doc in df[text_type+'_cut'].items()}
        np.save(r'model\vector_dict_' + text_type + '.npy', vector_dict)
        print("vector_dict词典已生成")

def text_diversity():
    # 读取csv文件
    df = pd.read_csv('data/visit_Q&A_cut.csv')
    # 设置“ReportID_Rank”为索引
    df['ReportID_Rank'] = df['ReportID'].astype(str) + '_' + df['Rank'].astype(str)
    df.set_index('ReportID_Rank', inplace=True)

    vector_dict = np.load('model/vector_dict_Q&A.npy', allow_pickle=True).item()

    calc_on_ReportID = True #False = calc on month
    if calc_on_ReportID == True:
        # 创建一个空的DataFrame，用于存储结果
        result_df = pd.DataFrame(columns=['ReportID','Diversity'])
        # 按照ReportID列分组，创建一个循环，遍历每个组
        for name, group in tqdm(df.groupby('ReportID')):
            # 计算组内各个向量的离散程度
            vectors = np.array([vector_dict[index] for index in group.index])
            mean_vector = np.mean(vectors, axis=0)
            euclidean_distances = np.mean([distance.euclidean(v, mean_vector) for v in vectors])
            # 将结果添加到result_df
            result_df = pd.concat([result_df, pd.DataFrame(
                {'ReportID': [name], 'Diversity': [euclidean_distances]})], ignore_index=True)
    else:
        # 创建一个空的DataFrame，用于存储结果
        result_df = pd.DataFrame(columns=['Symbol', 'ReportYear', 'ReportMonth', 'Diversity'])
        # 按照ReportID列分组，创建一个循环，遍历每个组
        for name, group in tqdm(df.groupby(['Symbol', 'ReportYear', 'ReportMonth'])):
            # 计算组内各个向量的离散程度
            vectors = np.array([vector_dict[index] for index in group.index])
            mean_vector = np.mean(vectors, axis=0)
            euclidean_distances = np.mean([distance.euclidean(v, mean_vector) for v in vectors])
            # 将结果添加到result_df
            result_df = pd.concat([result_df, pd.DataFrame(
                {'Symbol': [name[0]], 'ReportYear': [name[1]], 'ReportMonth': [name[2]],
                 'Diversity': [euclidean_distances]})], ignore_index=True)

    result_df.to_csv('data/reportID_diversity_Q&A.csv', encoding='utf-8-sig', index=False)

def relevance():
    df = pd.read_csv('data/Q&A_cut.csv')
    # 设置“ReportID_Rank”为索引
    df['ReportID_Rank'] = df['ReportID'].astype(str) + '_' + df['Rank'].astype(str)
    df.set_index('ReportID_Rank', inplace=True)

    # 2. 将分好词的文本转换为集合
    df['Question_set'] = df['Question_cut'].str.split().apply(set)
    df['Answer_set'] = df['Answer_cut'].str.split().apply(set)
    # 3. 创建ignore列
    def check_ignore(row):
        if len(row['Question_set'].intersection(row['Answer_set'])) == 0:
            return 1
        else:
            return 0
    df['ignore'] = df.apply(check_ignore, axis=1)
    # 4. 删除中间结果列
    df = df.drop(['Question_set', 'Answer_set'], axis=1)
    df = df[['ReportID','Symbol','ReportDate','ReportYear','Rank','Question','Answer','ignore']]
    filtered_df = df.groupby('ReportID').filter(lambda x: x['ignore'].eq(1).all()).reset_index()

    filtered_df.to_csv(r'data\relevance_sample.csv', encoding='utf-8-sig', index=False)


    # # 2. 使用Word2Vec模型将词语向量化
    # sentences = [text.split() for text in df['Question_cut'].tolist() + df['Answer_cut'].tolist()]
    # word2vec_model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, sg=0)  # 假设词向量维度为100
    # word2vec_model.save('model\word2vec')
    # print("word2vec模型训练完成")

    # 3. 计算词语相关性矩阵M
    word2vec_model = Word2Vec.load('model\word2vec')
    vocab = word2vec_model.wv.index_to_key
    M = np.zeros((len(vocab), len(vocab)))
    for i, word1 in enumerate(vocab):
        for j, word2 in enumerate(vocab):
            M[i, j] = word2vec_model.wv.similarity(word1, word2)

    result_df = pd.DataFrame(columns=['ReportID', 'Symbol', 'ReportYear', 'ReportMonth', 'Relevance'])
    for name, group in tqdm(df.groupby('ReportID')):
        sim_ls = []
        for index in group.index:
            # 4. 计算提问向量Q和回答向量A
            Q = np.array([word2vec_model.wv[word].count for word in df.loc[index, 'Question_cut'].str.split().explode().unique()])
            A = np.array([word2vec_model.wv[word].count for word in df.loc[index, 'Answer_cut'].str.split().explode().unique()])
            # 5. 计算软余弦相似比
            numerator = np.dot(Q, np.dot(M, A))
            denominator = np.sqrt(np.dot(Q, np.dot(M, Q)) * np.dot(A, np.dot(M, A)))
            sim = numerator / denominator
            sim_ls.append(sim)
        sim_mean = np.mean(sim_ls, axis=0)
        # 将结果添加到result_df
        result_df = pd.concat([result_df, pd.DataFrame(
            {'ReportID': [name], 'Symbol':[group['Symbol']], 'ReportYear':[group['ReportYear']], 'ReportMonth':[group['ReportMonth']], 'Relevance': [sim_mean]})], ignore_index=True)

    result_df.to_csv(r'data\reportID_relevance.csv', encoding='utf-8-sig', index=False)

    # relevance = pd.read_csv(r'data\reportID_relevance.csv')

    # # 读取实地调研数据
    # df = pd.read_csv('data/Q&A_cut.csv', usecols=['ReportID', 'Symbol', 'ReportYear', 'ReportMonth'])
    # relevance = relevance.merge(df, on=['ReportID'], how='left')
    # relevance.to_csv(r'data\reportID_relevance.csv', encoding='utf-8-sig', index=False)


def text_independence(inputpath, doc2vecpath):
    # 读取csv文件
    df = pd.read_csv(inputpath)
    # 设置“ReportID_Rank”为索引
    df['ReportID_Rank'] = df['ReportID'].astype(str) + '_' + df['Rank'].astype(str)

    # 使用apply方法对每个组进行操作
    vector_dict = np.load(doc2vecpath, allow_pickle=True).item()
    def compute_vector_mean(group):
        vectors = [vector_dict[report_id] for report_id in group['ReportID_Rank']]
        return np.mean(vectors, axis=0)
    calc_on_ReportID = True
    if calc_on_ReportID == True:
        grouped = df.groupby('ReportID').apply(compute_vector_mean)
        # 将结果保存到vector_mean_dict中
        vector_mean_dict = {}
        for reportID, vector in grouped.items():
            vector_mean_dict[reportID] = vector
        np.save('data/Q&Avector_mean_dict.npy', vector_mean_dict)
        print("vector_mean_dict词典已生成")
    else:
        grouped = df.groupby(['Symbol', 'ReportYear', 'ReportMonth']).apply(compute_vector_mean)
        # 将结果保存到vector_mean_dict中
        vector_mean_dict = {}
        for (symbol, year, month), vector in grouped.items():
            key = f"{symbol}_{year}_{month}"
            vector_mean_dict[key] = vector
        np.save('data/vector_mean_dict_Q&A.npy', vector_mean_dict)
        print("vector_mean_dict词典已生成")

    # 计算组内独立性
    independence_dict = {}
    firm_info=pd.read_excel('../firm_basicinfo.xlsx')
    for key in tqdm(vector_mean_dict.keys()):
        indus_code = firm_info.loc[firm_info['Symbol'].astype(str)  == key.split('_')[0],'Nnindcd'].str.cat()
        same_indus_symbol = list(firm_info[firm_info['Nnindcd']  == indus_code]['Symbol'].astype(str))
        same_indus_key = [symbol + "_" + key.split('_')[1] + "_" + key.split('_')[2] for symbol in same_indus_symbol]
        cos_sim = np.mean(
            [cosine_similarity([vector_mean_dict[key]], [vector_mean_dict[k]])[0][0] for k in same_indus_key if
             k != key and k in vector_mean_dict.keys()])
        independence_dict[key] = 1 - cos_sim
    # 创建dataframe
    result_df = pd.DataFrame(list(independence_dict.items()), columns=['key', 'Independence'])
    result_df['Symbol'] = result_df['key'].apply(lambda x: x.split('_')[0])
    result_df['ReportYear'] = result_df['key'].apply(lambda x: x.split('_')[1])
    result_df['ReportMonth'] = result_df['key'].apply(lambda x: x.split('_')[2])

    result_df = result_df[['Symbol', 'ReportYear', 'ReportMonth', 'Independence']]

    return result_df

def text_expert():
    df = pd.read_csv('data/visit_Q&A_cut.csv')
    calc_on_ReportID = True
    if calc_on_ReportID == True:
        grouped = df.groupby('ReportID')['Q&A'].apply(lambda x: ' '.join(x))
        number_ratio = grouped.apply(lambda x: sum(c.isdigit() for c in x) / len(x.strip()))
        result_df = number_ratio.reset_index()
        result_df.columns = ['ReportID', 'NumberRatio']
    else:
        grouped = df.groupby(['Symbol', 'ReportYear', 'ReportMonth'])['Q&A'].apply(lambda x: ' '.join(x))
        number_ratio = grouped.apply(lambda x: sum(c.isdigit() for c in x) / len(x.strip()))
        result_df = number_ratio.reset_index()
        result_df.columns = ['Symbol', 'ReportYear', 'ReportMonth', 'NumberRatio']

    result_df.to_csv('data/reportID_expert_Q&A.csv', encoding='utf-8-sig', index=False)

def text_sent():
    # 加载情感字典
    with open("../finsent_dict_RF.pkl",'rb') as f:
        finsent_dict=pickle.load(f)

    # 读取数据
    df = pd.read_csv('data/visit_Q&A_cut.csv')
    # 将Q&A_cut列中的浮点数值转换为字符串类型
    df['Q&A_cut'] = df['Q&A_cut'].astype(str)
    # 按ReportID分组，对组内Question列的文本进行拼接
    calc_on_ReportID = True
    if calc_on_ReportID == True:
        grouped = df.groupby('ReportID')['Q&A_cut'].apply(lambda x: ' '.join(x))
    else:
        grouped = df.groupby(['Symbol', 'ReportYear', 'ReportMonth'])['Q&A_cut'].apply(lambda x: ' '.join(x))
    grouped = grouped.reset_index()
    # 统计词汇数的函数
    def count_words(text):
        try:
            words = text.split()  # 使用空格分割词汇
        except:
            words = []
        return len(words)
    # 统计每个单元格的词汇数
    grouped['WordCount'] = grouped['Q&A_cut'].apply(count_words)
    print("已计算文本词汇个数")

    pos_count_ls, neg_count_ls = [], []
    for text in tqdm(grouped['Q&A_cut']):
        text = text.replace(" ","")
        pos_count = 0
        neg_count = 0
        for word in finsent_dict['positive']:
            pos_count += text.count(word)
        for word in finsent_dict['negative']:
            neg_count += text.count(word)
        pos_count_ls.append(pos_count)
        neg_count_ls.append(neg_count)

    grouped['pos_count'] = pos_count_ls
    grouped['neg_count'] = neg_count_ls

    grouped['PosRatio'] = grouped['pos_count']/grouped['WordCount']
    grouped['NegRatio'] = grouped['neg_count'] / grouped['WordCount']
    grouped['Sent'] = grouped['PosRatio'] - grouped['NegRatio']

    if calc_on_ReportID == True:
        result_df = grouped[['ReportID', 'PosRatio', 'NegRatio', 'Sent']]
    else:
        result_df = grouped[['Symbol', 'ReportYear', 'ReportMonth', 'PosRatio', 'NegRatio', 'Sent']]

    result_df.to_csv('data/reportID_sent_Q&A.csv', encoding='utf-8-sig', index=False)

def obj_dict_gen():
    xls = pd.ExcelFile('../金融情感词典RF.xlsx')

    positive_words = pd.read_excel(xls, 'positive', usecols=[0]).values.flatten().tolist()
    negative_words = pd.read_excel(xls, 'negative', usecols=[0]).values.flatten().tolist()

    with open('../extreme_words.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    strong_words = [line.strip() for line in lines]

    extreme_words = []
    for word in positive_words + negative_words:
        for adv in strong_words:
            if adv in word:
                extreme_words.append(word)

    extreme_words = list(set(extreme_words))
    # 创建字典
    words_dict = {'positive': positive_words, 'negative': negative_words, 'extreme': extreme_words}
    # 保存为pkl文件
    with open('finsent_dict_RF.pkl', 'wb') as f:
        pickle.dump(words_dict, f)

def text_obj():
    # obj_dict_gen()
    # 加载情感字典
    with open("../finsent_dict_RF.pkl", 'rb') as f:
        finsent_dict = pickle.load(f)
    # extreme_words = " ".join(finsent_dict['extreme'])

    # 读取数据
    df = pd.read_csv('data/visit_Q&A_cut.csv')
    # 将Q&A_cut列中的浮点数值转换为字符串类型
    df['Q&A_cut'] = df['Q&A_cut'].astype(str)
    calc_on_ReportID = True
    if calc_on_ReportID == True:
        grouped = df.groupby('ReportID')['Q&A'].apply(lambda x: ''.join(x))
        grouped = grouped.reset_index()
        obj_scores = []
        for text in tqdm(grouped['Q&A']):
            extrm_count = 0
            sent_count = 0
            for word in finsent_dict['extreme']:
                extrm_count += text.count(word)
            for word in finsent_dict['positive'] + finsent_dict['negative']:
                sent_count += text.count(word)
            if sent_count != 0:
                obj_scores.append(1 - extrm_count / sent_count)
            else:
                obj_scores.append(1)

        grouped['Obj'] = obj_scores
        result_df = grouped[['ReportID', 'Obj']]
    else:
        grouped = df.groupby(['Symbol', 'ReportYear', 'ReportMonth'])['Q&A'].apply(lambda x: ''.join(x))
        grouped = grouped.reset_index()
        obj_scores = []
        for text in tqdm(grouped['Q&A']):
            extrm_count = 0
            sent_count = 0
            for word in finsent_dict['extreme']:
                extrm_count += text.count(word)
            for word in finsent_dict['positive'] + finsent_dict['negative']:
                sent_count += text.count(word)
            if sent_count != 0:
                obj_scores.append(1 - extrm_count / sent_count)
            else:
                obj_scores.append(1)

        grouped['Obj'] = obj_scores
        result_df = grouped[['Symbol', 'ReportYear', 'ReportMonth', 'Obj']]

    result_df.to_csv('data/reportID_obj_Q&A.csv', encoding='utf-8-sig', index=False)


def text_activity(inputpath1, inputpath2):
    # 1. 读取visit_Q&A.csv并按照指定列分组求Q&A列的数据个数
    qa_df = pd.read_csv(inputpath1)
    qa_grouped = qa_df.groupby(['Symbol', 'ReportYear', 'ReportMonth']).size().reset_index(name='Q&A_count')
    # 2. 读取visit_investor.csv并按照指定列分组求institutionName列的数据个数
    investor_df = pd.read_csv(inputpath2)
    investor_grouped = investor_df.groupby(['Symbol', 'ReportYear', 'ReportMonth']).size().reset_index(
        name='investor_count')
    # 3. 合并两张表的分组结果
    merged_df = pd.merge(qa_grouped, investor_grouped, on=['Symbol', 'ReportYear', 'ReportMonth'], how='inner').fillna(
        0)
    # 4. 计算Activity列
    merged_df['Activity'] = merged_df['Q&A_count'] / merged_df['investor_count']
    result_df = merged_df[['Symbol', 'ReportYear', 'ReportMonth', 'Activity']]

    return result_df

def text_readable():

    ADV_words = pickle.load(open('../ADV_CONJ.pkl', 'rb'))['ADV']
    CONJ_words = pickle.load(open('../ADV_CONJ.pkl', 'rb'))['CONJ']

    adv_conj_words = set(ADV_words + CONJ_words)

    # 读取数据
    df = pd.read_csv('data/visit_Q&A_cut.csv')


    def readable_calc(text):
        zi_num_per_sent = []
        adv_conj_ratio_per_sent = []
        text = re.sub('\d+\.\d+|\.\d+', 'num', text)
        # 【分句】
        def cn_seg_sent(text):
            # split the chinese text into sentences
            text = re.sub('([。！；？;\?])([^”’])', "[[end]]", text)  # 单字符断句符
            text = re.sub('([。！？\?][”’])([^，。！？\?])', "[[end]]", text)
            text = re.sub('\s', '', text)
            # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
            return text.split("[[end]]")

        sentences = cn_seg_sent(text)
        for sent in sentences:
            adv_conj_num = 0
            zi_num_per_sent.append(len(sent))
            words = list(jieba.cut(sent))
            for w in words:
                if w in adv_conj_words:
                    adv_conj_num += 1
            adv_conj_ratio_per_sent.append(adv_conj_num / (len(words) + 1))

        def standardize(list):
            mean = np.mean(list)
            std_dev = np.std(list)
            normalized_list = [(x - mean) / std_dev for x in list]

            return normalized_list

        readability1 = np.mean([np.log(x+1) for x in zi_num_per_sent])
        readability2 = np.mean([np.log(x+1) for x in adv_conj_ratio_per_sent])
        readability3 = 1/((readability1 + readability2) * 0.5)

        return readability3

    calc_on_ReportID = True
    if calc_on_ReportID == True:
        # 按ReportID分组，对组内Question列的文本进行拼接
        grouped = df.groupby('ReportID')['Q&A'].apply(lambda x: ''.join(x))
        grouped = grouped.reset_index()
        grouped['Readable'] = grouped['Q&A'].apply(readable_calc)
        result_df = grouped[['ReportID', 'Readable']]
    else:
        grouped = df.groupby('ReportID')['Q&A'].apply(lambda x: ''.join(x))
        grouped = grouped.reset_index()
        grouped['Readable'] = grouped['Q&A'].apply(readable_calc)
        result_df = grouped[['ReportID', 'Readable']]

    result_df.to_csv('data/reportID_spread_Q&A.csv', encoding='utf-8-sig', index=False)

def freq_alter(csv_file, type):

    # 定义一个函数来转换月份为季度
    def month_to_quarter(month):
        if month in [1, 2, 3]:
            return 1
        elif month in [4, 5, 6]:
            return 2
        elif month in [7, 8, 9]:
            return 3
        else:
            return 4

    # 读取csv文件
    data = pd.read_csv(os.path.join('output2', csv_file))

    if type == 'quarterly':
        # 添加ReportQuarter列
        data['ReportQuarter'] = data['ReportMonth'].apply(month_to_quarter)

        # 将ReportYear和ReportQuarter组合为一个新的列，用于分组
        data['YearQuarter'] = data['ReportYear'].astype(str) + 'Q' + data['ReportQuarter'].astype(str)

        # 对YearQuarter进行分组，取第一个数据和最后一列的均值
        target_column = data.columns[-3]
        grouped = data.groupby(['Symbol', 'YearQuarter']).agg({col: 'first' for col in data.columns if col != target_column})
        grouped[target_column] = data.groupby(['Symbol', 'YearQuarter'])[target_column].mean()

        # 删除YearQuarter列
        grouped = grouped.drop(columns=['YearQuarter'])
        grouped = grouped[['Symbol', 'ReportYear', 'ReportQuarter',target_column]]

    if type == 'yearly':
        # 对YearQuarter进行分组，取第一个数据和最后一列的均值
        target_column = data.columns[-1]
        grouped = data.groupby(['Symbol', 'ReportYear']).agg(
            {col: 'first' for col in data.columns if col != target_column})
        grouped[target_column] = data.groupby(['Symbol', 'ReportYear'])[target_column].mean()
        grouped = grouped[['Symbol', 'ReportYear', target_column]]

    result_df = grouped.reset_index(drop=True)

    return result_df

def sum_stat(files):
    VstDvrsty = pd.read_csv('output2/'+ files[0])['Diversity']
    VstIndpn = pd.read_csv('output2/'+ files[1])['Independence']
    VstExprt = pd.read_csv('output2/'+ files[2])['NumberRatio']
    VstObj = pd.read_csv('output2/'+ files[3])['Obj']
    VstSprd = pd.read_csv('output2/'+ files[4])['Readable']
    VstSent = pd.read_csv('output2/' + files[5])['Sent']

    VstQlty = 0.2*(VstDvrsty/VstDvrsty.std() + VstIndpn/VstIndpn.std() + VstExprt/VstExprt.std() + VstObj/VstObj.std() + VstSprd/VstSprd.std())

    sum_stat = pd.DataFrame({'VstDvrsty':VstDvrsty.describe(),
                             'VstIndpn':VstIndpn.describe(),
                             'VstExprt':VstExprt.describe(),
                             'VstObj': VstObj.describe(),
                             'VstSprd':VstSprd.describe(),
                             'VstQlty': VstQlty.describe(),
                             'VstSent': VstSent.describe()})

    # 构建数据框
    data = pd.DataFrame({
        'VstDvrsty': VstDvrsty,
        'VstIndpn': VstIndpn,
        'VstExprt': VstExprt,
        'VstObj': VstObj,
        'VstSprd': VstSprd,
        'VstQlty': VstQlty
    })

    # 计算相关系数矩阵
    correlation_matrix = data.corr()

    # 打印相关系数矩阵
    print(correlation_matrix)

    return sum_stat


if __name__ == '__main__':
    # data_clear()
    text_cut()
    # doc2vec()

    # relevance()
    # df=pd.read_csv(r'data\reportID_relevance.csv')
    # result = df.groupby(['Symbol', 'ReportYear'])['Relevance'].agg(['max', 'min']).reset_index()
    # result['Difference'] = result['max'] - result['min']
    # result.to_csv('data/relevance_stat.csv', encoding='utf-8-sig', index=False)

    # inputpath = 'input2/visit_Q&A_cut.csv'
    # result_df = text_diversity(inputpath)
    # result_df.to_csv('output2/Q&A_diversity_byID.csv', encoding='utf-8-sig',index=False)
    # df1 = pd.read_csv('output2/Q&A_diversity_byID.csv')
    # df2 = pd.read_csv('input2/visit_title.csv',usecols=['ReportID','ReportDate','Symbol','Title'])
    # df3 = pd.read_csv('input2/visit_basic.csv',usecols=['ReportID','MeetingWay','Researcher'])
    # df = pd.merge(df1,df2, on='ReportID', how='left')
    # df = pd.merge(df, df3, on='ReportID', how='left')
    # df.to_csv('output2/Q&A_diversity_example.csv', encoding='utf-8-sig',index=False)

    # inputpath = 'input2/visit_Q&A_cut.csv'
    # doc2vecpath = 'data/vector_dict_Q&A.npy'
    # result_df = text_independence(inputpath, doc2vecpath)
    # result_df.to_csv('output2/text_independence_Q&A.csv', encoding='utf-8-sig', index=False)

    # text_diversity()
    # text_expert()
    # text_obj()
    # text_readable()
    # text_sent()

    # df1 = pd.read_csv(r"D:\python_work\corporate_site_visits\data\text_independence_answer.csv")
    # df2 = pd.read_csv(r"D:\python_work\corporate_site_visits\data\text_independence_question.csv")
    # df = df1.merge(df2, on='ReportID')
    # df['Independence'] = (df['Independence_x'] + df['Independence_x'])/2
    # df = df[['ReportID','Independence']]
    # df.to_csv(r"data\text_independence_Q&A.csv")

    # inputpath1 = 'input2/visit_Q&A.csv'
    # inputpath2 = 'input2/visit_investor.csv'
    # result_df = text_activity(inputpath1, inputpath2)
    # result_df.to_csv('output2/text_activity.csv', encoding='utf-8-sig', index=False)

    # inputpath = 'input2/visit_Q&A.csv'
    # result_df = text_readable(inputpath)
    # result_df.to_csv('output2/text_read_Q&A.csv', encoding='utf-8-sig', index=False)

    # files = ['text_diversity_Q&A.csv', 'text_independence_Q&A.csv', 'text_expert_Q&A.csv', 'text_sent_Q&A.csv', 'text_activity.csv', 'text_obj_Q&A.csv', 'text_read_Q&A.csv']
    # for file in files:
    #     df = freq_alter(file, 'yearly')
    #     df.to_csv(os.path.join('output2', 'yearly' + file), index = False)

    # files = ['text_diversity_Q&A.csv', 'text_independence_Q&A.csv', 'text_expert_Q&A.csv',
    #          'text_obj_Q&A.csv', 'text_read_Q&A.csv', 'text_sent_Q&A.csv']
    # files_quarterly = ['quarterly' + file for file in files]
    # files_yearly = ['yearly' + file for file in files]
    # sum_stat = sum_stat(files)
    # sum_stat.to_csv('output2/sum_stat.csv', encoding='utf-8-sig', index=False)