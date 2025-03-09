# Self Attention Mechanism 
```ruby
   class SelfAttention(nn.Module):
    def __init__(self, embedding_dims, heads, dropout):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.embedding_dims = embedding_dims
        self.head_dims = int(embedding_dims / heads)

        self.key = nn.Linear(self.head_dims, self.head_dims , bias = False)
        self.query = nn.Linear(self.head_dims, self.head_dims , bias = False)
        self.value = nn.Linear(self.head_dims, self.head_dims , bias = False)

        self.fc = nn.Linear(self.head_dims * self.heads, self.embedding_dims)
        self.dropout = nn.Dropout(dropout)
```
# 
