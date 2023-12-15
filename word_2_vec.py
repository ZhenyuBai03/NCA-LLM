import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

class CBOW(nn.Module):

    def __init__(self, context_size=2, embedding_size=100, vocab_size=None):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        lookup_embeds = self.embeddings(inputs)
        embeds = lookup_embeds.sum(dim=0)
        out = self.linear1(embeds)
        out = F.log_softmax(out)
        return out



# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# print(make_context_vector(data[0][0], word_to_ix))  # example

if __name__ == '__main__':
    CONTEXT_SIZE = 1  # 2 words to the left, 2 to the right
    EMBEDDING_SIZE = 10
    raw_text = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules
    called a program. People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells.""".split()

    # By deriving a set from `raw_text`, we deduplicate the array
    vocab = set(raw_text)
    vocab_size = len(vocab)
    print(vocab)

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}
    data = []
    
    for index in range(len(raw_text)-CONTEXT_SIZE):
        context = raw_text[index:index+CONTEXT_SIZE*2+1]
        target = context.pop(CONTEXT_SIZE)
        data.append((context, target))

    loss_func = nn.CrossEntropyLoss()
    net = CBOW(CONTEXT_SIZE, embedding_size=EMBEDDING_SIZE, vocab_size=vocab_size)
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(500):
        total_loss = 0
        for context, target in data:
            context_var = make_context_vector(context, word_to_ix)
            net.zero_grad()
            log_probs = net(context_var)

            loss = loss_func(log_probs.view(1,-1), autograd.Variable(
                torch.LongTensor([word_to_ix[target]])
            ))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(total_loss)
