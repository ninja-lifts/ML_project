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
```ruby
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
    attention_score = torch.softmax(attention_score, dim=-1)

    out = torch.einsum('bhqv,bvhd->bqhd', [attention_score, value]).reshape(
        Batch, query_len, self.heads * self.head_dims
    )

    out = self.dropout(self.fc(out))
    return out



         

