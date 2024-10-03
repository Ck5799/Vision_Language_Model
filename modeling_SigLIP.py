from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:  #PaliGemma comes in different sizes so each has a different config we are buliding 224 X 224

    def __init__( 
            self,  
            hidden_size=768,
            intermediate_size=3072, 
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=16,
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens: int =None,
            **kwargs         
            ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens 

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config =SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(         # convolution is done 
            in_channels=config.num_channels, # number of colours
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,  # size of the patch to traverse
            stride=self.patch_size,       # amount of traversal
            padding="valid", # no padding
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2  #pow of 2 beacuse img is 2d
        self.num_positions = self.num_patches  # encode info about where it came from
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim) # learned embedding
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1,-1)),
            persistent=False,
        )

    def forward(self, pixle_values: torch.FloatTensor)->torch.Tensor:
        _,_,height,width = pixle_values.shape  #input using numpy which is in [batch_size, Channel, Height, width]
        #resize image to 224
        #convlove the image 

        patch_embeds = self.patch_embedding(pixle_values) # extracting this pathc embedding to convolution 

        embeddings = patch_embeds.flatten(2) # flattening it

        embeddings = embeddings.transpose(1,2) # we want the number of patches to come before the embed_dim

        embeddings = embeddings + self.position_embedding(self.position_ids) # add positional encoding to embeddings

        return embeddings
    

class SiglipAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # equivalent to 1/ sqrt(self.head_dim)
        self.dropout = config.attention_dropout


        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor,Optional[torch.Tensor]]:
        
        batch_size, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states) 
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        attn_weights = (torch.matmul(query_states,key_states.transpose(2,3))*self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size,self.num_heads,seq_len,seq_len)},but is"
                f"{attn_weights.size()}"
            )
        
        #apply Softmax
        attn_weights=nn.functional.softmax(attn_weights,dim=-1, dtype=torch.float32).to(query_states.dtype)
        #apply dropout
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
       
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention Output should be of size {(batch_size,self.num_heads,seq_len,self.head_dim)},but is"
                f"{attn_output.size()}"
            )
        attn_output=attn_output.transpose(1,2).contiguous()
        attn_output=attn_output.reshape(batch_size, seq_len, self.embed_dim)

        return attn_output






class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor)->torch.Tensor:
        # convert the embed_dim to intermediate size
        hidden_states = self.fc1(hidden_states)
        # apply GELU function
        hidden_states=nn.functional.gelu(hidden_states, approximate="tanh")
        # second layer get back to the embed_dim 
        hidden_states = self.fc2(hidden_states)
        return hidden_states



class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(
            self,
            hidden_states: torch.Tensor
    )-> torch.Tensor:
        # save the input (residual connection ) as it is a skip connecton and we need to use it later
        residual = hidden_states
        # run the input through the layer normalization not changing the shape of input
        #[batch_size, Num_patches, Embed_Dim]->[Batch_size, Num_patches, Embed_dim]---> change in values only
        hidden_states = self.layer_norm1(hidden_states)
        # next allpy self attention
        # [batch_size, Num_patches, Embed_Dim]-> [batch_size, Num_patches, Embed_Dim]
        hidden_states, _ =self.self_attn(hidden_states=hidden_states)
        # adding the residual connection
        hidden_states=residual+hidden_states
        # now the output is again stored and is anther skip connection
        residual= hidden_states
        # another layer normalization 
        hidden_states= self.layer_norm2(hidden_states)
        #sending it to MLP
        hidden_states=self.mlp(hidden_states)
        # another addition of skip connection
        hidden_states=residual+hidden_states

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
            self, 
            input_embeds: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = input_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config) # extracts patches from the image
        self.encoder = SiglipEncoder(config) # extraced patches are run throught the layers of transformer
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps) # normalization of the layer

    def forward(self,pixel_values:torch.Tensor)->torch.Tensor:
     
        # we will take the pixle values ( patch of images ) and  get the embeddings after running it through convolution (done by SiglipVisionEmbeddings) 
        # now we take this embeddings(patches + positional encoding) andd run it through the encoder
        # for vision transformer the normalization is done before the feed forward and the multi head attention

        hidden_states = self.embeddings(pixel_values)

        last_hidden_states = self.encoder(input_embeds=hidden_states)

        last_hidden_states = self.post_layernorm(last_hidden_states)

        return last_hidden_states

class SiglipVisonModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values)->Tuple:

        # we will load the image with numpy which will convert it into an array like below:
        # [batch_size, Channels, Height, width] -> [batch_size, Num_Patches, Embed_Dim] 
        # which is then converted by the vision transformer into a list of embeddings  
        # for each image and each embedding of size Embed_Dim 

        return self.vision_model(pixel_values=pixel_values)


