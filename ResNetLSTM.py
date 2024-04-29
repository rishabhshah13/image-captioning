import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from Dataloader import FlickrDataset,CapsCollate
import torchvision.transforms as T
# Import argparse for hyperparameters
import argparse
import os

# Define and parse hyperparameters
parser = argparse.ArgumentParser(description='Image Captioning Hyperparameters')
parser.add_argument('--embed_size', type=int, default=400, help='Size of word embeddings')
parser.add_argument('--hidden_size', type=int, default=512, help='Size of hidden units in LSTM')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in LSTM')
parser.add_argument('--drop_prob', type=float, default=0.3, help='Dropout probability')
args = parser.parse_args()

# Hyperparameters
embed_size = args.embed_size
hidden_size = args.hidden_size
learning_rate = args.learning_rate
num_epochs = args.num_epochs
num_layers = args.num_layers
drop_prob = args.drop_prob

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#setting the constants
data_location =  "../input/flickr8k"
caption_file = os.path.join(data_location, 'captions.txt')
BATCH_SIZE = 256
NUM_WORKER = 2

#defining the transform to be applied
transforms = T.Compose([
    T.Resize(256),                     
    T.RandomCrop(224),                 
    T.ToTensor(),                               
    T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

dataset = FlickrDataset(
    root_dir=data_location + "/Images",
    captions_file=caption_file,
    transform=transforms
)

BATCH_SIZE = 256
NUM_WORKER = 2

pad_idx = dataset.vocab.stoi["<PAD>"]

data_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    shuffle=True,
    collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
)

# Define Encoder CNN
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

# Define Decoder RNN
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fcn = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(drop_prob)
    
    def forward(self, features, captions):
        embeds = self.embedding(captions[:, :-1])
        x = torch.cat((features.unsqueeze(1), embeds), dim=1) 
        x, _ = self.lstm(x)
        x = self.fcn(x)
        return x
    
    def generate_caption(self, inputs, hidden=None, max_len=20, vocab=None):
        batch_size = inputs.size(0)
        captions = []
        
        for i in range(max_len):
            output, hidden = self.lstm(inputs, hidden)
            output = self.fcn(output)
            output = output.view(batch_size, -1)
        
            predicted_word_idx = output.argmax(dim=1)
            captions.append(predicted_word_idx.item())
            
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break
            
            inputs = self.embedding(predicted_word_idx.unsqueeze(0))
        
        return [vocab.itos[idx] for idx in captions]
        
class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, drop_prob)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


#helper function to save the model
def save_model(model,num_epochs):
    model_state = {
        'num_epochs':num_epochs,
        'embed_size':embed_size,
        'vocab_size':len(dataset.vocab),
        'state_dict':model.state_dict()
    }

    torch.save(model_state,'model_state.pth')


# Initialize model, loss, etc.
model = EncoderDecoder(embed_size, hidden_size, len(dataset.vocab), num_layers, drop_prob).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print_every = 100

for epoch in range(1, num_epochs + 1):   
    for idx, (image, captions) in enumerate(iter(data_loader)):
        image, captions = image.to(device), captions.to(device)
        optimizer.zero_grad()
        outputs = model(image, captions)
        loss = criterion(outputs.view(-1, len(dataset.vocab)), captions.view(-1))
        loss.backward()
        optimizer.step()
        
        if (idx + 1) % print_every == 0:
            print("Epoch: {} Loss: {:.5f}".format(epoch, loss.item()))
            
            model.eval()
            with torch.no_grad():
                dataiter = iter(data_loader)
                img, _ = next(dataiter)
                features = model.encoder(img[0:1].to(device))
                caps = model.decoder.generate_caption(features.unsqueeze(0), vocab=dataset.vocab)
                caption = ' '.join(caps)
                # show_image(img[0], title=caption)
                
            model.train()
        
    save_model(model,epoch)
