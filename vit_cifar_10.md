```ruby
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
```
# Self Attention Mechanism 
```ruby
class SelfAttention(nn.Module): # This defines a class SelfAttention, which inherits from torch.nn.Module
  def __init__(self, embedding_dims, heads, dropout):
        #__init__ is the constructor that initializes the parameters.
        # embedding_dims: The size of the input embedding (total feature dimensions).
        # heads: The number of attention heads (splitting embeddings into multiple attention subspaces).
        # dropout: A dropout rate for regularization to prevent overfitting 
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.embedding_dims = embedding_dims
        self.head_dims = int(embedding_dims / heads)

        self.key = nn.Linear(self.head_dims, self.head_dims , bias = False)
        self.query = nn.Linear(self.head_dims, self.head_dims , bias = False)
        self.value = nn.Linear(self.head_dims, self.head_dims , bias = False)
        # nn.Linear(in_features, out_features, bias): A fully connected linear transformation layer

        self.fc = nn.Linear(self.head_dims * self.heads, self.embedding_dims)
        self.dropout = nn.Dropout(dropout)

  def forward(self, query, key, value, mask):
    Batch = query.shape[0]
    query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]

    query = query.reshape(Batch, query_len, self.heads, self.head_dims)
    key = key.reshape(Batch, key_len, self.heads, self.head_dims)
    value = value.reshape(Batch, value_len, self.heads, self.head_dims)

    query = self.query(query)
    key = self.key(key)
    value = self.value(value)

    attention_score = torch.einsum('bqhd,bkhd->bhqk', [query, key])

    if mask is not None:
        attention_score = attention_score.masked_fill(mask == 0, float('-1e20'))

    attention_score = attention_score / ((self.head_dims) ** (1/2))
      # Normalizes the attention scores by dividing by 𝑑𝑘 (square root of head dimensions). Prevents large values from 
      # causing instability in softmax.
    attention_score = torch.softmax(attention_score, dim=-1)

    out = torch.einsum('bhqv,bvhd->bqhd', [attention_score, value]).reshape(
        Batch, query_len, self.heads * self.head_dims
    )

    out = self.dropout(self.fc(out))
    return out
```
# Transformer block
```ruby
 class TransformerBlock(nn.Module):
    def __init__(self, embedding_dims, heads, dropout, forward_expansion, layer_norm_eps):
        # layer_norm_eps: Small value added to avoid division by zero in LayerNorm.
        super(TransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.attention = SelfAttention(embedding_dims, heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dims, embedding_dims * forward_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dims * forward_expansion, embedding_dims),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
       #   Define input tensor (batch_size=2, seq_length=5, embedding_dim=128 , Expected output: torch.Size([2, 5, 128]) 

        norm = self.layer_norm1(x)
        attention_block = self.attention(norm, norm, norm, mask)
        add = x + attention_block
        norm = self.layer_norm2(add)
        feed_forward = self.feed_forward(norm)
        out = feed_forward + add
        return out
```
# Vit Model
```ruby
 class ViT(nn.Module):
    def __init__(
        self, patch_height, patch_width, max_len, embedding_dims, heads,
        forward_expansion, num_layers, dropout, layer_norm_eps, num_classes
    ):
        super(ViT, self).__init__()

        self.vit_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dims,
                    heads,
                    dropout,
                    forward_expansion,
                    layer_norm_eps
                )
                for _ in range(num_layers)
            ]
        )
        # self.vit_blocks = nn.Sequential(
        #   TransformerBlock(...),
        #   TransformerBlock(...),
        #   TransformerBlock(...),
        #   TransformerBlock(...)
)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, embedding_dims))
        self.patch_embeddings = nn.Linear(embedding_dims, embedding_dims)
        self.postional_embedding = nn.Parameter(torch.zeros(1, max_len + 1, embedding_dims))
        self.to_cls_token = nn.Identity()
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dims),
            nn.Linear(embedding_dims, num_classes * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_classes * 4, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
```
# Splitting image into patches then patch , positional embedding and classfication embeddding
```ruby
   def forward(self, images):   # image size = [batch_size, channels, height, width] , eg. -> (N , 3 , 32 , 32)
        patches = images.unfold(2, self.patch_height, self.patch_width).unfold(3, self.patch_height, self.patch_width)  # # 
              # Extracting small patches , extracting small patches along dim = 2(height) , extracting along width(dim = 3)   
        patches = patches.permute(0, 2, 3, 1, 4, 5) # rearanging patches so easy use later in reshapping .
         # 0 → batch size (unchanged).
         # 2,3 → number of patches along height and width.
         # 1 → the number of channels.
         # 4,5 → patch height and width.

        patches = patches.reshape(
            patches.shape[0],
            patches.shape[1],
            patches.shape[2],
            patches.shape[3] * patches.shape[4] * patches.shape[5]
        ) # Now the patches tensor has shape [batch_size, num_patches_height, num_patches_width, flattened_patch_size]

        patches = patches.view(patches.shape[0], -1, patches.shape[-1])
        x = self.cls_embedding.expand(patches.shape[0], -1, -1)
          # The class (CLS) token is a special trainable vector that acts as a summary representation of the entire image.
        patch_embeddings = self.patch_embeddings(patches)
          # This step converts the raw patch vectors into meaningful feature representations.
        x = torch.cat((x, patch_embeddings), dim=1) + self.postional_embedding
        x = self.dropout(x)
        mask = None
        for block in self.vit_blocks:
            x = block(x, mask)
        out = self.to_cls_token(x[:, 0])
        out = self.classifier(out) # This is the final classification layer (probably a fully connected layer with softmax activation).
        return out  # This layer produces the final class scores.
```
# Training on Cifar-10
```ruby
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = CIFAR10(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = CIFAR10(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = ViT(
    patch_height=16,
    patch_width=16,
    embedding_dims=768,
    dropout=0.1,
    heads=4,
    num_layers=6,
    forward_expansion=4,
    max_len=int((32 * 32) / (16 * 16)),
    layer_norm_eps=1e-5,
    num_classes=10,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

```


         

