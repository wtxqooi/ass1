# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pickle

import torch

# from Ass1.training_model import encode_and_add_padding_and_unknown


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi("fasdfsa")
    i=1
    para = eval("".join(open("para" + str(i) + ".txt", "r").readlines()))
    # testing_data = pickle.load(open("testing_data.pkl", "rb"))
    # doc_list = []
    # label_list = []
    # for emo, context in testing_data:
    #     doc_list.append(context)
    #     label_list.append(1 if emo == "pos" else 0)
    # maxlength = para["seq_max_len"]
    # sent_encoded = encode_and_add_padding_and_unknown(doc_list, maxlength)
    # filename = str(para["curr_file"]) + "_attempt_.pt"
    # model = torch.load(filename)
    # model.eval()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
