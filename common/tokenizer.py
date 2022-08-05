MAX_VOCAB_SIZE = 30000
UNK, PAD = '<UNK>', '<PAD>'


class Segmentation:
    def __init__(self):
        """
        :param vocab: 词典
        """
        # self.vocab = vocab
        self.vocab = {}
        self.tokenize = lambda x: x.strip(' ').split(' ')

    def max_match_forword(self, input_sentence, max_len: int):
        result = []
        tmp = 0
        while tmp < len(input_sentence):
            for i in range(max_len):
                index = tmp + max_len - i
                if input_sentence[tmp: index] in self.vocab:
                    result.append(input_sentence[tmp:index])
                    tmp = index
                    break
                elif index == 1 + tmp:
                    result.append(input_sentence[tmp])
                    tmp = index
        return result

    def build_vocab(self,file_name, max_size, min_freq):
        vocab_dic = {}
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                lin = line.strip()
                if not lin:
                    continue
                content = lin.split('\t')[0]
                # print(content)

                for word in self.tokenize(content):
                    # print(word)
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1
                # break
            vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                         :max_size]
            vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        self.vocab = vocab_dic.keys()
        print(vocab_dic)
        print(vocab_list)

    def load_vocab():
        pass


if __name__ == '__main__':
    # import jieba
    # jieba.add_word('⾃然语⾔处理')
    # i = jieba.cut_for_search('⾃然语⾔处理是计算机科学领域与⼈⼯智能领域中的⼀个重要⽅向')
    # print('\',\''.join(i))
    vocab = ['⾃然', '语⾔', '⾃然语⾔', '处理', '计算', '算机', '科学', '计算机', '计算机科学', '领域', '⼈⼯', '智能', '⼈⼯智能', '领域', '⼀个']
    model = Segmentation()
    # sentence = '⾃然语⾔处理是计算机科学领域与⼈⼯智能领域中的⼀个重要⽅向'
    model.build_vocab('../data/train_tgt.txt', max_size=MAX_VOCAB_SIZE, min_freq=1)
    # result = model.max_match_forword(sentence, max_len=7)
    print(0)
