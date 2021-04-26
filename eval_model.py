import pickle
import numpy as np
import torch
from gensim.models import Word2Vec, FastText
from nltk import word_tokenize

from Ass1.training_model import encode_and_add_padding_and_unknown, Bi_RNN_Model, clean_word_list

testing_data = pickle.load(open("testing_data.pkl", "rb"))


# Prediction
def eval_model(para):
    doc_list = []
    label_list = []
    for emo, context in testing_data:
        doc_list.append(clean_word_list(context))
        label_list.append(1 if emo == "pos" else 0)
    maxlength = para["seq_max_len"]
    curr_vector = get_vector(para)
    sent_encoded = encode_and_add_padding_and_unknown(doc_list, maxlength, curr_vector)
    filename = str(para["curr_file"]) + "_attempt_.pt"
    model = torch.load(filename)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model(torch.from_numpy(sent_encoded).int().to(device))
    predicted = torch.argmax(outputs, 1)
    # print(outputs[:10])
    # print(predicted[:10])
    from sklearn.metrics import classification_report
    print("Frame:", para["frame"], end="\t")
    print("Emb_model:", para["emb_model"])
    print("n_input:", para["n_input"], end="\t")
    print("n_class:", para["n_class"], end="\t")
    print("n_hidden:", para["n_hidden"])
    print("batch_size:", para["batch_size"], end="\t")
    print("total_epoch:", para["total_epoch"], end="\t")
    print("learning_rate:", para["learning_rate"])
    print(classification_report(np.array(label_list), predicted.cpu().numpy(), digits=4))


def eval_vectors(wv, vocab, prefix='./question_file'):
    """Evaluate the trained word vectors on a variety of tasks"""

    filenames = [
        'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
        'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
        'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
        'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
        'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt',
    ]

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 1000

    correct_sem = 0  # count correct semantic questions
    correct_syn = 0  # count correct syntactic questions
    correct_tot = 0  # count correct questions
    count_sem = 0  # count all semantic questions
    count_syn = 0  # count all syntactic questions
    count_tot = 0  # count all questions
    full_count = 0  # count all questions, including those with unknown words

    for i in range(len(filenames)):
        with open('%s/%s' % (prefix, filenames[i]), 'r') as f:
            full_data = [line.rstrip().split(' ') for line in f]
            full_count += len(full_data)
            data = [x for x in full_data if all(word in vocab for word in x)]

        if len(data) == 0:
            continue

        indices = np.array([[vocab[word] for word in row] for row in data])
        ind1, ind2, ind3, ind4 = indices.T

        predictions = np.zeros((len(indices),))
        num_iter = int(np.ceil(len(indices) / float(split_size)))
        for j in range(num_iter):
            subset = np.arange(j * split_size, min((j + 1) * split_size, len(ind1)))

            pred_vec = (wv.vectors[ind2[subset], :] - wv.vectors[ind1[subset], :]
                        + wv.vectors[ind3[subset], :])

            # cosine similarity if input W has been normalized
            dist = np.dot(wv.vectors, pred_vec.T)

            for k in range(len(subset)):
                dist[ind1[subset[k]], k] = -np.Inf
                dist[ind2[subset[k]], k] = -np.Inf
                dist[ind3[subset[k]], k] = -np.Inf

            # predicted word index
            predictions[subset] = np.argmax(dist, 0).flatten()

        val = (ind4 == predictions)  # correct predictions
        count_tot = count_tot + len(ind1)
        correct_tot = correct_tot + sum(val)
        if i < 5:
            count_sem = count_sem + len(ind1)
            correct_sem = correct_sem + sum(val)
        else:
            count_syn = count_syn + len(ind1)
            correct_syn = correct_syn + sum(val)
    return correct_sem, correct_syn, correct_tot, count_sem, count_syn, count_tot, full_count


def get_vector(para):
    if para["frame"] == "Word2Vec":
        curr_vector = Word2Vec.load(para["file_name"]).wv
    else:
        curr_vector = FastText.load(para["file_name"]).wv
    return curr_vector


if __name__ == '__main__':
    for i in range(1, 10):
        # try:
            curr_para = eval("".join(open(str(i) + "_para" + ".txt", "r").readlines()))
            curr_vector = get_vector(curr_para)
            vocab = {w: i for i, w in enumerate(curr_vector.index_to_key)}

            correct_sem, correct_syn, correct_tot, count_sem, count_syn, count_tot, full_count = \
                eval_vectors(curr_vector, vocab=vocab)
            print('Semantic accuracy: %.2f%%  (%i/%i)' %
                  (100 * correct_sem / float(count_sem), correct_sem, count_sem))
            print('Syntactic accuracy: %.2f%%  (%i/%i)' %
                  (100 * correct_syn / float(count_syn), correct_syn, count_syn))
            print('Total accuracy: %.2f%%  (%i/%i)' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))
            eval_model(curr_para)
            print("*" * 50)
        # except Exception as e:
        #     # print(e)
        #     print("eval ending")
        #     break
