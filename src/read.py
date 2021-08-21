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
    def __init__(self, num_characters_in, vocab_size, embedding_size):
        super().__init__()
        self.num_characters_in = num_characters_in
        self.embed = t.nn.Embedding(vocab_size, embedding_size)
        self.w1 = t.nn.Linear(num_characters_in, 100)
        self.w2 = t.nn.Linear(100, 100)
        self.w3 = t.nn.Linear(100, vocab_size)
        self.softmax = t.nn.Softmax(dim=1)

    def forward(self, character_indexes):
        assert len(character_indexes) == self.num_characters_in
        x = self.embed(character_indexes)
        x = t.unsqueeze(t.squeeze(x, dim=1), dim=0) # why?
        x = self.w1(x)
        x = self.w2(x)
        x = self.w3(x)
        x = self.softmax(x)
        return x

def sample_character(softmax):
    x = t.multinomial(softmax, num_samples=1)
    return INDEX_TO_C[x.item()]


with open('brief.txt', 'r') as f:
    raw_dataset = f.read()

tokens = []
for i in range(0, len(raw_dataset)):
    tokens.append(t.LongTensor([char_to_index(raw_dataset[i])]))
tokens = t.cat(tokens)
assert tokens[0] == tokens[1]
assert tokens[0] != tokens[2]

PAGE_SIZE = 10
predictor = CharacterPredictor(PAGE_SIZE, len(C_TO_INDEX), 1)

start_index = 2
o = predictor(tokens[start_index:start_index+PAGE_SIZE])
one_hot_label = t.zeros([len(C_TO_INDEX)])
one_hot_label[tokens[start_index+PAGE_SIZE+1]] = 1
loss = t.sum(t.abs(o - one_hot_label))

print(sample_character(o))
print(sample_character(one_hot_label))
print(loss)

exit(0)

for i in range(0, input_embedding.shape[0] - PAGE_SIZE):
    embeddings = input_embedding[i:i+PAGE_SIZE]
    print(embeddings)
    print(embeddings.shape)
    break
