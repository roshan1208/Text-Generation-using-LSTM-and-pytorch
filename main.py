# Library
import torch
import torch.nn as nn
import os
import numpy as np
from torch.nn.utils import clip_grad_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU set-up

# creating dictionary 
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}    # word to index, key: word; value: index
        self.idx2word = {}    # index to word, key: index; value:word
        self.idx = 0
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __len__(self):
        return len(self.word2idx)


# Text Proccessing
class TextProcess(object):
    def __init__(self):
        self.dictionary = Dictionary()
        
    def get_data(self, path, batch_size = 20):
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
                
        # Create 1D tensor that contains the index of all the words in the file
        rep_tensor = torch.LongTensor(tokens)
        index = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    rep_tensor[index] = self.dictionary.word2idx[word]
                    index += 1
        # find out how many batch we need
        num_batches = rep_tensor.shape[0] // batch_size
        # remove the remainder (filter out the ones that don't fit)
        rep_tensor = rep_tensor[:num_batches * batch_size]
        rep_tensor = rep_tensor.view(batch_size, -1)   # (batch_size, num_batches)
        return rep_tensor



# ************   change here For new data set, put .txt dataset in same folder *********************
# **************************************************************************************************

# Defining Parameter 
embedd_size = 128    # word is getting embedding 128 dimension vector
hidden_size = 1024    # hidden neural of each layer
num_layers = 2     # Number of LSTM layer
num_epochs = 100   # Number of training epochs
batch_size = 20
timesteps = 30      # Consider 30 timestep to predict next word
learning_rate = 0.002 
path = 'alice.txt'  # Corpus Dataset path

# *************************************************************************************************
# ****************   only change here   ***********************************************************



# create corpus
corpus = TextProcess()

# set represented Tensor, vocabulary Size and Number of Batchees
rep_tensor = corpus.get_data(path, batch_size)
vocab_size = len(corpus.dictionary)
num_batches = rep_tensor.shape[1] // timesteps

print('Batch size shape: {}'.format(rep_tensor.shape))
print('Vocabulary size: {}'.format(vocab_size))
print('Number of batches: {}'.format(num_batches))


# LSTM model 
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedd_size, hidden_size, num_layers):
        super(TextGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedd_size)    # word transfer to 5290*128 vector
        self.lstm = nn.LSTM(embedd_size, hidden_size, num_layers, batch_first=True)
#         self.linear1 = nn.Linear(hidden_size, hidden_size)
#         self.drop = nn.Dropout(0.2)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        
        
    def forward(self, x, h):
        # perform word embedding
        x = self.embed(x)
        # x = x.view(batch_size, timesteps, embedd_size)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))  # (batch_size*timesteps, hidden_size)
#         out = self.linear1(out)
#         out = self.drop(out)
        out = self.linear2(out)
        return out, (h, c)



# Load model
model = TextGenerator(vocab_size, embedd_size, hidden_size, num_layers).to(device)
# loss function
loss_fn = nn.CrossEntropyLoss()
#optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training the network
model.train()
for epoch in range(num_epochs):
    # set initial hidden and cell state
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
             torch.zeros(num_layers, batch_size, hidden_size).to(device))
#     states = states.to(device)
    
    for i in range(0, rep_tensor.size(1) - timesteps, timesteps):
        # get mini-batch input and targets
        inputs = rep_tensor[:, i:i + timesteps].to(device)
        targets = rep_tensor[:, (i+1):(i+1) + timesteps].to(device)
        
        #example sentence: ram is outstanding: 
        # input = ram is
        # output = am is o
        outputs, _ = model(inputs, states)
        loss = loss_fn(outputs, targets.reshape(-1))
        
        model.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()
        
        step = (i + 1) // timesteps
        
        if step % 100 == 0:
            print('Epoch [{}/{}]; Loss: {:.3f}'.format(epoch+1, num_epochs, loss.item()))



#Testing and Generating new Text of same corpus
model.eval()
with torch.no_grad():
    with open('results.txt', 'w') as f:
        states = (torch.zeros(num_layers, 1, hidden_size).to(device),
             torch.zeros(num_layers, 1, hidden_size).to(device))
        
        inputs = torch.randint(0, vocab_size, (1,)).long().unsqueeze(1).to(device)
        for i in range(1000):
            output, _ = model(inputs, states)
#             print(output.shape)
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()
#             print(word_id)
            inputs.fill_(word_id)

            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word +  ' '
            f.write(word)

#             if (i+1)%100 == 0:
#                 print('Sample: [{}/{}] word and save to {}'.format(i+1, 500, 'result.txt'))
                
                
with open('results.txt', 'r') as f:
    for line in f:
        print(line)