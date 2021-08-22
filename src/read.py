import torch as t
import torch.nn.functional as F
import random

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
        self.w1 = t.nn.Linear(num_characters_in * embedding_size, 100)
        self.w2 = t.nn.Linear(100, 100)
        self.w3 = t.nn.Linear(100, vocab_size)
        self.softmax = t.nn.Softmax(dim=1)

    def forward(self, character_indexes, softmax=False):
        assert character_indexes.shape[-1] == self.num_characters_in
        x = self.embed(character_indexes).view((-1, self.w1.in_features))
        x = F.relu(self.w1(x))
        x = F.relu(self.w2(x))
        x = F.relu(self.w3(x))
        if softmax:
            x = self.softmax(x)
        return x

def sample_character(y):
    i = None
    if isinstance(y, t.Tensor) and list(y.shape) == []:
        i = int(y.item())
    elif isinstance(y, int):
        i = y
    if isinstance(i, int):
        y = t.zeros([len(C_TO_INDEX)])
        y[i] = 1
    x = t.multinomial(y, num_samples=1)
    return INDEX_TO_C[x.item()]


with open('brief.txt', 'r') as f:
    raw_dataset = f.read()
    raw_dataset = raw_dataset.replace('\n', ' ')
    # raw_dataset = raw_dataset[:1000]

tokens = []
for i in range(0, len(raw_dataset)):
    tokens.append(t.LongTensor([char_to_index(raw_dataset[i])]))
tokens = t.cat(tokens)
assert tokens[0] == tokens[1]
assert tokens[0] != tokens[2]

PAGE_SIZE = 100
def load_batch(batch_size):
    batch_X = []
    batch_Y = t.zeros([batch_size], dtype=t.long)
    for i in range(0, batch_size):
        start_index = random.randint(0, len(tokens) - PAGE_SIZE - 1)
        batch_X.append(tokens[start_index:start_index+PAGE_SIZE])
        batch_Y[i] = tokens[start_index+PAGE_SIZE]
    return t.stack(batch_X), batch_Y

print('Some example data...')
batch_X, batch_Y = load_batch(13)
for i in range(0, len(batch_X)):
    x = ''.join([INDEX_TO_C[z.item()] for z in batch_X[i]])
    y = sample_character(batch_Y[i])
    print('{}[{}]'.format(x, y))
print()

print('batch_X shape = {}'.format(batch_X.shape))

cross_entropy = t.nn.CrossEntropyLoss()
predictor = CharacterPredictor(PAGE_SIZE, len(C_TO_INDEX), 1)
optimizer = t.optim.Adam(predictor.parameters(), lr=1e-3)
for i in range(0, 1000000):
    batch_X, batch_Y = load_batch(15)
    o = predictor(batch_X)
    loss = cross_entropy(o, batch_Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(loss.item())
        sample_size = 2
        sample_batch, _ = load_batch(sample_size)
        o = predictor(sample_batch, softmax=True)
        for i in range(0, sample_size):
            x = ''.join([INDEX_TO_C[z.item()] for z in sample_batch[i]])
            y = sample_character(o[i])
            print('{}[{}]'.format(x, y))
        print()
