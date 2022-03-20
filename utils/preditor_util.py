from model import RN
import cv2
import string
import numpy as np
import torch
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP'], default='RN', help='resume from model stored')
parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str, help='resume from model stored')
parser.add_argument('--relation-type', type=str, default='binary', help='what kind of relations to learn. options: binary, ternary (default: binary)')

args = parser.parse_args()
torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args.cuda = False

class RNPredictor():
    def __init__(self):
        pass

    def __tensor_data(self, data):
        image, question = data
        image = torch.FloatTensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2)
        question = torch.FloatTensor([question])
        return image, question

    def __map_qustion_type(self, idx):
        question_idx = 12
        question_sub_type = 15
        if idx == 1:
            question_idx += 0
            question_sub_type += 0
        elif idx == 2:  
            question_sub_type += 1
        elif idx == 3:
            question_sub_type += 2
        elif idx == 4:
            question_idx += 1
            question_sub_type += 0
        elif idx == 5:
            question_idx += 1
            question_sub_type += 1
        elif idx == 6:
            question_idx += 1
            question_sub_type += 2
        return question_idx, question_sub_type

    def __convert_to_vector(self, question_idx, tokens):
        color_map = { 'red':0, 'green':1, 'blue':2, 'orange':3, 'gray':4, 'yellow':5 }
        question = [ 0 for i in range(18) ]
        for word in tokens:
            if word in color_map.keys():
                question[color_map[word]] = 1
        question_idx, question_sub_type = self.__map_qustion_type(question_idx)
        # print(question_idx, question_sub_type)
        question[question_idx] = 1
        question[question_sub_type] = 1
        return question

    def tokenize(self, sentence):
        type_split = sentence.split('.')
        question_idx = int(type_split[0])
        sentence = type_split[1]
        for c in sentence:
            if c in string.punctuation:
                no_punctuation_sentence = sentence.replace(c, '') 
        tokens = no_punctuation_sentence.split()
        question = self.__convert_to_vector(question_idx, tokens)
        # print(question)
        return question

    def predict(self, data):
        classes = [ 'yes', 'no', 'rectangle', 'circle', '1', '2', '3', '4', '5', '6' ]
        model = RN(args)
        checkpoint = torch.load('best_model/epoch_RN_20.pth')
        model.load_state_dict(checkpoint)
        model.eval()
        input_image, input_question = self.__tensor_data(data)
        output = model(input_image, input_question)
        output = F.log_softmax(output, dim=0)
        _, prediction = output.data.max(0)
        # print(_, prediction)
        return classes[prediction]
        
'''
# testing
if __name__ == '__main__':
    image = cv2.imread('image_RGB.jpg')
    colors = [ 'red', 'green', 'blue', 'orange', 'gray', 'yellow' ]

    questions = [
    "1.What is the shape of the <color> object ?",
    "2.Is <color> object placed on the left side of the image ?",
    "3.Is <color> object placed on the upside of the image ?",
    "4.What is the shape of the object closest to the <color> object ?",
    "5.What is the shape of the object furthest to the <color> object ?",
    "6.How many objects have same shape with the <color> object ?" ]

    rn_predictor = RNPredictor()
    for quetion in questions:
        for color in colors:
            sentence = quetion.replace('<color>', color)
            print('Question:', sentence)
            question = rn_predictor.tokenize(sentence)
            result = rn_predictor.predict((image/255, question))
            print('Answer:', result)
'''