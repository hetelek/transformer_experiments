import torch as t
from torch.autograd import Variable

C_COUNT = 0
C_TO_INDEX = {}
INDEX_TO_C = {}
def char_to_index(c):
    global C_COUNT, C_TO_EMBEDDING
    index = C_TO_INDEX.get(c)
    if index is None:
        index = C_COUNT
        C_COUNT += 1
        C_TO_INDEX[c] = index
        INDEX_TO_C[index] = c
    return index

def index_to_char(i):
    return C_TO_INDEX[i]

class CharacterPredictor(t.nn.Module):
    def __init__(self, num_embeddings, embedding_size):
        self.embed = t.nn.Embedding(num_embeddings, embedding_size)
        self.w1 = t.nn.Linear(embedding_size, 100)
        self.w2 = t.nn.Linear(100, 100)
        self.w3 = t.nn.Linear(100, embedding_size)
        self.softmax = t.nn.Softmax(dim=None)

    def forward(self, character_indexes):
        x = self.embed(character_indexes)
        x = self.w1(x)
        x = self.w2(x)
        x = self.w3(x)
        x = self.softmax(x)
        return x


with open('brief.txt', 'r') as f:
    raw_dataset = f.read()

tokens = []
for i in range(0, len(raw_dataset)):
    tokens.append(t.LongTensor([char_to_index(raw_dataset[i])]))
tokens = t.cat(tokens)
assert tokens[0] == tokens[1]
assert tokens[0] != tokens[2]

embedding = t.nn.Embedding(len(C_TO_INDEX), 1)
input_embedding = embedding(tokens)
assert input_embedding[0] == input_embedding[1]
assert input_embedding[0] != input_embedding[2]


PAGE_SIZE = 10
for i in range(0, input_embedding.shape[0] - PAGE_SIZE):
    embeddings = input_embedding[i:i+PAGE_SIZE]
    print(embeddings)
    print(embeddings.shape)
    break
