import random
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# STEP 6: Test the model
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
patterns = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        patterns.append((w, tag))



ignore_words = ['?', '!', '.']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))


# TBD: Load the intents file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# TBD: Retrieve the model and all the sizings

# TBD: build the NN
data = torch.load("trained.pth")

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# TBD: prepare a command-line conversation (don't forget something to make the user exit the script!)
botname = 'siri'
print('lets chat')

while True:
    sentence = input('you : ')
    if sentence == "exit":
        print("thanks for joining me!")
        break
    else:
        #print('you: ' + sentence)
        pass
    
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)

    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs =torch.softmax(output, dim =1)
    prob = probs[0][predicted.item()]
    #print(prob)
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:

                print(f"{botname}: {random.choice(intent['responses'])}")

                if random.choice(intent['responses']) == "would you like to schedule an appointment?":
                    response = input('you : ')
                    if response == ('yes' or 'Yes' or 'ok' or 'sure'):
                        print(f"{botname}: When do you want to schedule the appointment to?")
                        response = input('you : ')
                        print(f"{botname}: Appointment scheduled for - {response}")
                    else:
                        print(f"{botname}: how else can i help you then?")
                        break
                else:
                    break

    else:
                print(f"{botname}: i do not understand. try to be more specific pls")