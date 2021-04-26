import pickle
from random import shuffle
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import re

training_data = pickle.load(open("training_data.pkl", "rb"))


def clean_word_list(string):
    # print(string)
    from nltk.corpus import stopwords
    contraction_dict = np.load("contraction.npy", allow_pickle=True).item()

    def _emoji(matched) -> str:
        temp = matched.group(0).replace(" ", "")
        if temp[1].isalpha():
            temp = temp[0:2]
        elif temp[1].isdigit():
            temp = ' '
        else:
            temp = temp[0:3]
        return " " + temp + " "

    def _braket(matched) -> str:
        temp = matched.group(0)
        return temp[1:-1]

    string = re.sub(r"#\w*|&\w*|@\w*", " [unknown] ", string)
    string = re.sub(r"\r|\n", " ", string)
    string = re.sub(r"http://[^\s]+\s?|https://[^\s]+\s?", " ", string)
    string = re.sub(r"[^A-Za-z()<>:?\'\[\]]",  " ", string)
    string = re.sub(r"\([\S\s]*\)", _braket, string)
    string = re.sub(r"\?+", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\.{2,}", " ... ", string)
    string = re.sub(r":(\s)*[\S]+", _emoji, string)
    string = string.lower().split(" ")
    temp_result = []
    stop_words = stopwords.words("english")
    for s in string:
        try:
            temp = contraction_dict[s].split(" ")
            temp_result = temp_result + temp
        except KeyError:
            temp_result.append(s)
    string = temp_result

    string = [f for f in string if f not in stop_words and f != ""]
    return string


def encode_and_add_padding_and_unknown(sentences, seq_length, vector):
    word_dict = {w: i for i, w in enumerate(vector.index_to_key)}
    sent_encoded = []
    for sent in sentences:
        encode_doc = []
        for word in sent:
            try:
                temp_encoded = word_dict[word.lower()]
                encode_doc.append(temp_encoded)
            except KeyError:
                encode_doc.append(word_dict["[unknown]"])
        if len(encode_doc) < seq_length:
            encode_doc += [word_dict['[padding]']] * (seq_length - len(encode_doc))
        else:
            encode_doc = encode_doc[:seq_length]
        sent_encoded.append(encode_doc)
    # print(sent_encoded)
    return np.array(sent_encoded)


def get_word_embedding(docs, n_input, max_len):
    def _check_lexicon(word:str) -> float:
        neg = open("negative-words.txt", "r", encoding="GBK").readlines()[31:]
        pos = neg = open("positive-words.txt", "r", encoding="GBK").readlines()[31:]
        neg = [w.strip("\n") for w in neg]
        pos = [w.strip("\n") for w in pos]
        if word in pos:
            return 0.5
        elif word in neg:
            return 0.0
        else:
            return -0.5

    def _init_word_embedding(data, dataset_name: str, modeltype: str, sg, vector_size=24, window=3):
        frame = "Skip_Gram" if sg == 1 else "CBoW"
        para["file_name"] = modeltype + "_" + dataset_name + "_" + frame + "_" + str(vector_size + 1) + "_" + str(
            window)
        if modeltype == "FastText":
            from gensim.models import FastText
            try:
                embedding_model = FastText.load(para["file_name"])
            except:
                embedding_model = FastText(data, vector_size=vector_size, window=window, min_count=5, workers=4, sg=sg,epochs=50)

        else:
            from gensim.models import Word2Vec
            try:
                embedding_model = Word2Vec.load(para["file_name"])
            except:
                embedding_model = Word2Vec(data, vector_size=vector_size, window=window, min_count=5, workers=4, sg=sg,epochs=50)
        embedding_model.save(para["file_name"])
        return embedding_model

    # print(len(docs))
    for i in range(len(docs)):
        if len(docs[i]) < max_len:
            temp = docs[i]
            docs[i] = temp + ['[padding]'] * (max_len - len(docs[i]))
    dim = n_input - 1
    win = 6
    trained_wv = _init_word_embedding(docs, "nps+webtext+training", frame, 1 if emb_model == "skip_gram" else 0, dim,
                                      win).wv
    embedded_word = []
    for i in range(len(trained_wv.vectors)):
        temp = np.append(trained_wv.vectors[i], _check_lexicon(trained_wv.index_to_key[i]))
        embedded_word.append(temp)
    embedded_word = np.array(embedded_word)
    # np.save(str(para["curr_file"]) + "_emb_word.npy", embedded_word)
    return trained_wv, embedded_word


class Bi_RNN_Model(nn.Module):
    def __init__(self):
        super(Bi_RNN_Model, self).__init__()
        self.emb = nn.Embedding(vocab_size, n_input)
        self.emb.weight.data.copy_(torch.from_numpy(word_embedding))
        self.rnn = nn.LSTM(n_input, n_hidden, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        self.linear = nn.Linear(n_hidden * 2, n_class)
        self.emb.weight.requires_grad = False

    def forward(self, x):
        x = self.emb(x)
        # print(x.shape)
        result, (h_n, c_n) = self.rnn(x)
        hidden_out = torch.cat((h_n[0, :, :], h_n[1, :, :]), 1)
        output = self.linear(hidden_out)
        return output


para = eval("".join(open("para.txt", "r").readlines()))
emb_model = para["emb_model"]
frame = para["frame"]
n_input = para["n_input"]
n_class = para["n_class"]
n_hidden = para["n_hidden"]
batch_size = para["batch_size"]
total_epoch = para["total_epoch"]
learning_rate = para["learning_rate"]
if __name__ == '__main__':
    doc_list = []
    label_list = []
    for emo, context in training_data:
        doc_list.append(clean_word_list(context))
        label_list.append(1 if emo == "pos" else 0)
    label_list = np.array(label_list)
    length_list = []
    for d in doc_list:
        length_list.append(len(d))
    seq_max_len = max(length_list)
    wv, word_embedding = get_word_embedding(doc_list, n_input, seq_max_len)
    print(word_embedding.shape)
    sent_encoded = encode_and_add_padding_and_unknown(doc_list, seq_max_len, wv)
    vocab_size = len(word_embedding)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Bi_RNN_Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(total_epoch):
        train_loss = 0
        for ind in range(0, sent_encoded.shape[0], batch_size):
            input_batch = np.array(sent_encoded[ind:min(ind + batch_size, sent_encoded.shape[0])])
            target_batch = np.array(label_list[ind:min(ind + batch_size, sent_encoded.shape[0])])
            input_batch_torch = torch.from_numpy(input_batch).to(device).int()
            target_batch_torch = torch.from_numpy(target_batch).view(-1).to(device).long()
            # ------------------------------------------------
            model.train()
            optimizer.zero_grad()
            outputs = model(input_batch_torch)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_batch_torch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # -------------------------------------------------
        print('Epoch: %d, train loss: %.5f' % (epoch + 1, train_loss))

    print('Finished Training')
    curr_file = para["curr_file"]
    # print("save word vector")
    # np.save(str(curr_file) + "_vector.npy", wv)
    # print("finish save word vector")
    parafile = str(para["curr_file"]) + "_para" + ".txt"
    para["seq_max_len"] = seq_max_len

    print("save para")
    open(parafile, "w").write(str(para))
    para["curr_file"] = para["curr_file"] + 1
    print("update para")
    open("para.txt", "w").write(str(para))
    print("save model")
    torch.save(model, str(curr_file) + "_attempt_" + '.pt')
    print("process end")
