import numpy as np
import datetime
import random


class DataSet():
    def __init__(self, filename):
        self.filename = filename
        self.sentences = []
        self.tags = []
        sentence = []
        tag = []
        word_num = 0
        file = open(filename, encoding='utf-8')
        while True:
            line = file.readline()
            if not line:
                break
            if line == '\n':
                self.sentences.append(sentence)  # [[word1,word2,...],[word1...],[...]]
                self.tags.append(tag)  # [[tag1,tag2,...],[tag1...],[...]]
                sentence = []
                tag = []
            else:
                sentence.append(line.split()[1])  # [word1,word2,...]
                tag.append(line.split()[3])  # [tag1,tag2,...]
                word_num += 1
        self.sentences_num = len(self.sentences)  # 统计句子个数
        self.word_num = word_num  # 统计词语个数

        print('{}:共{}个句子,共{}个词。'.format(filename, self.sentences_num, self.word_num))
        file.close()

    def split(self):
        data = []
        temp_list = []
        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):  # j为词在句子中的序号
                temp_list.append((self.sentences[i][j], self.tags[i][j]))  # [(句子1,1,词性),(句子1,2,词性)...]
            data.append(temp_list)
            temp_list = []
        return data


class global_liner_model(object):
    def __init__(self, train_file, dev_file, test_file):
        self.train_data = DataSet(train_file)  # 处理训练集文件
        self.dev_data = DataSet(dev_file)  # 处理开发集文件
        self.test_data = DataSet(test_file) # 处理测试集文件
        self.features = {}  # 存放所有特征及其对应编号的字典
        self.index_to_tag = {}  # 存放所有词性及其对应编号的字典
        self.tag_list = []  # 存放所有词性的列表
        self.weights = []  # 特征权重矩阵
        self.v = []
        self.BOS = 'BOS'

    def create_bigram_feature(self, pre_tag, cur_tag):
        return ['01:' + cur_tag + '*' + pre_tag]

    def create_unigram_feature(self, sentence, position, cur_tag):
        template = []
        cur_word = sentence[position]
        cur_word_first_char = cur_word[0]
        cur_word_last_char = cur_word[-1]
        if position == 0:
            last_word = '##'
            last_word_last_char = '#'
        else:
            last_word = sentence[position - 1]
            last_word_last_char = sentence[position - 1][-1]

        if position == len(sentence) - 1:
            next_word = '$$'
            next_word_first_char = '$'
        else:
            next_word = sentence[position + 1]
            next_word_first_char = sentence[position + 1][0]

        template.append('02:' + cur_tag + '*' + cur_word)
        template.append('03:' + cur_tag + '*' + last_word)
        template.append('04:' + cur_tag + '*' + next_word)
        template.append('05:' + cur_tag + '*' + cur_word + '*' + last_word_last_char)
        template.append('06:' + cur_tag + '*' + cur_word + '*' + next_word_first_char)
        template.append('07:' + cur_tag + '*' + cur_word_first_char)
        template.append('08:' + cur_tag + '*' + cur_word_last_char)

        for i in range(1, len(sentence[position]) - 1):
            template.append('09:' + cur_tag + '*' + sentence[position][i])
            template.append('10:' + cur_tag + '*' + sentence[position][0] + '*' + sentence[position][i])
            template.append('11:' + cur_tag + '*' + sentence[position][-1] + '*' + sentence[position][i])
            if sentence[position][i] == sentence[position][i + 1]:
                template.append('13:' + cur_tag + '*' + sentence[position][i] + '*' + 'consecutive')

        if len(sentence[position]) > 1 and sentence[position][0] == sentence[position][1]:
            template.append('13:' + cur_tag + '*' + sentence[position][0] + '*' + 'consecutive')

        if len(sentence[position]) == 1:
            template.append('12:' + cur_tag + '*' + cur_word + '*' + last_word_last_char + '*' + next_word_first_char)

        for i in range(0, 4):
            if i > len(sentence[position]) - 1:
                break
            template.append('14:' + cur_tag + '*' + sentence[position][0:i + 1])
            template.append('15:' + cur_tag + '*' + sentence[position][-(i + 1)::])
        return template

    def create_feature_template(self, sentence, position, pre_tag, cur_tag):
        template = []
        template.extend(self.create_bigram_feature(pre_tag, cur_tag))
        template.extend(self.create_unigram_feature(sentence, position, cur_tag))
        return template

    def create_feature_space(self):
        for i in range(len(self.train_data.sentences)):
            sentence = self.train_data.sentences[i]
            tags = self.train_data.tags[i]  # 当前句子的分词
            for j in range(len(sentence)):
                if j == 0:
                    pre_tag = self.BOS
                else:
                    pre_tag = tags[j - 1]
                template = self.create_feature_template(sentence, j, pre_tag, tags[j])
                for f in template:  # 对特征进行遍历
                    if f not in self.features.keys():  # 如果特征不在特征字典中，则添加进去
                        self.features[f] = len(self.features)  # 给该特征一个独立的序号标记
                for tag in tags:
                    if tag not in self.tag_list:
                        self.tag_list.append(tag)
        self.tag_list = sorted(self.tag_list)
        self.index_to_tag = {i: t for i, t in enumerate(self.tag_list)}  # 分别列出词性下标和词性
        self.weights = np.zeros(len(self.features))
        self.v = np.zeros(len(self.features))
        self.update_times = np.zeros(len(self.features))
        self.bigram_features = [
            [self.create_bigram_feature(prev_tag, tag) for prev_tag in self.tag_list]
            for tag in self.tag_list
        ]
        print("特征的总数是：{}".format(len(self.features)))

    def get_score(self, feature, averaged):
        score = 0
        for f in feature:
            if f in self.features:
                if averaged:
                    score += self.v[self.features[f]]
                else:
                    score += self.weights[self.features[f]]
        return score

    def predict(self, sentence, averaged):
        length = len(sentence)
        delta = np.zeros((length, len(self.tag_list)))
        paths = np.zeros((length, len(self.tag_list)), dtype='int')
        delta[0] = np.array([self.get_score(self.create_feature_template(sentence, 0, self.BOS, cur_tag), averaged)
                             for cur_tag in self.tag_list])
        paths[0] = -1

        bigram_scores = np.array(
            [[self.get_score(bigram_f, averaged) for bigram_f in bigram_feature] for bigram_feature in
             self.bigram_features])  # [[tag1_tag1,tag2_tag1,...],[tag1_tag2,tag2_tag2,...],...,] tag*tag
        for i in range(1, length):  # 对每一个词
            unigram_scores = np.array(
                [self.get_score(self.create_unigram_feature(sentence, i, tag), averaged) for tag in
                 self.tag_list])  # [x,x,x,...x] tag个score
            scores = bigram_scores + unigram_scores[:, None] + delta[i - 1]  # none用于升维
            # 求出每一行(即每个cur_tag)的最大索引，保存到path[i]:[a b c....]
            paths[i] = np.argmax(scores, axis=1)
            # 记录下每一行(即每个cur_tag)的最大分数
            delta[i] = np.max(scores, axis=1)
        predict_tag_list = list()
        tag_index = np.argmax(delta[length - 1])  # 对最后一个词的tag，寻找它最可能的前一个tag在paths中的索引
        predict_tag_list.append(self.index_to_tag[tag_index])
        for i in range(length - 1):
            tag_index = paths[length - 1 - i][tag_index]  # paths中的索引即为真正的tag
            predict_tag_list.insert(0, self.index_to_tag[tag_index])
        return predict_tag_list

    def update_v(self, index, update_time, last_w_value):
        last_update_time = self.update_times[index]  # 上一次更新所在的次数
        current_update_time = update_time  # 本次更新所在的次数
        self.update_times[index] = update_time  # 修改为本次更新所在的次数
        self.v[index] += (current_update_time - last_update_time - 1) * last_w_value + self.weights[index]

    def evaluate(self, data, averaged):
        total_num = 0
        correct_num = 0
        for i in range(len(data.sentences)):
            sentence = data.sentences[i]
            tags = data.tags[i]
            total_num += len(tags)
            predict = self.predict(sentence, averaged)
            for j in range(len(tags)):
                if tags[j] == predict[j]:
                    correct_num += 1
        return correct_num, total_num, correct_num / total_num

    def perceptron_online_training(self, iterator, stop_iterator, shuffle, averaged):
        max_dev_precision = 0
        counter = 0
        update_time = 0
        max_iterator = -1
        data = self.train_data.split()
        if averaged:
            print("使用累加特征权重：")
        else:
            print("不使用累加特征权重：")
        for iter in range(iterator):
            print('iterator: {}'.format(iter))
            start_time = datetime.datetime.now()
            if shuffle:
                print('\t正在打乱训练数据...')
                random.shuffle(data)
                print("\t数据已打乱，正在进行预测...")
            for i in range(len(self.train_data.sentences)):
                sentence = self.train_data.sentences[i]
                tags = self.train_data.tags[i]
                predict = self.predict(sentence, averaged)
                if predict != tags:
                    update_time += 1
                    for j in range(len(tags)):
                        if j == 0:
                            gold_pre_tag = self.BOS
                            predict_pre_tag = self.BOS
                        else:
                            gold_pre_tag = tags[j - 1]
                            predict_pre_tag = predict[j - 1]
                        gold_feature = self.create_feature_template(sentence, j, gold_pre_tag, tags[j])
                        predict_feature = self.create_feature_template(sentence, j, predict_pre_tag, predict[j])
                        for f in gold_feature:
                            if f in self.features:
                                findex = self.features[f]
                                last_w_value = self.weights[findex]
                                self.weights[self.features[f]] += 1
                                self.update_v(findex, update_time, last_w_value)

                        for f in predict_feature:
                            if f in self.features:
                                findex = self.features[f]
                                last_w_value = self.weights[findex]
                                self.weights[self.features[f]] -= 1
                                self.update_v(findex, update_time, last_w_value)
            # 本次迭代完成
            current_update_times = update_time  # 本次更新所在的次数
            for i in range(len(self.v)):
                last_w_value = self.weights[i]
                last_update_times = self.update_times[i]  # 上一次更新所在的次数
                if current_update_times != last_update_times:
                    self.update_times[i] = current_update_times
                    self.v[i] += (current_update_times - last_update_times - 1) * last_w_value + self.weights[i]

            train_correct_num, total_num, train_precision = self.evaluate(self.train_data, averaged)
            print('\t' + 'train准确率：{} / {} = {}'.format(train_correct_num, total_num, train_precision))
            test_correct_num, test_num, test_precision = self.evaluate(self.test_data, averaged)
            print('\t' + 'test准确率：{} / {} = {}'.format(test_correct_num, test_num, test_precision))
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev_data, averaged)
            print('\t' + 'dev准确率：{} / {} = {}'.format(dev_correct_num, dev_num, dev_precision))

            if dev_precision > max_dev_precision:
                max_dev_precision = dev_precision
                max_iterator = iter
                counter = 0
            else:
                counter += 1
            end_time = datetime.datetime.now()
            print("\t迭代执行时间为：" + str((end_time - start_time).seconds) + " s")
            if counter >= stop_iterator:
                break
        print('最优迭代轮次 = {} , 开发集准确率 = {}'.format(max_iterator, max_dev_precision))


if __name__ == '__main__':
    train_data_file = 'data/train.conll'  # 训练集文件
    dev_data_file = 'data/dev.conll'  # 开发集文件
    test_data_file = 'data/test.conll'  # 测试集文件
    iterator = 100  # 最大迭代次数
    stop_iterator = 10  # 连续多少次迭代没有提升效果就退出
    shuffle = True  # 每次迭代是否打乱数据
    averaged = False  # 是否使用averaged percetron

    total_start_time = datetime.datetime.now()
    lm = global_liner_model(train_data_file, dev_data_file, test_data_file)
    lm.create_feature_space()
    lm.perceptron_online_training(iterator, stop_iterator, shuffle, averaged)
    total_end_time = datetime.datetime.now()
    print("总执行时间为：" + str((total_end_time - total_start_time).seconds) + " s")
