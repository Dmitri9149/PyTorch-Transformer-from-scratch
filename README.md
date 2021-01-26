### PyTorch-Transformer-from-scratch
PyTorch Transformer architecture from scratch

The Transformer (see article "Attention is All You Need" : https://arxiv.org/abs/1706.03762 ) is used here for machine 
translation and new architecture experimentation. The code is in PyTorch.

I start with basic code from d2l.ai project: http://d2l.ai/index.html ; https://github.com/d2l-ai/d2l-en ;   

@book{zhang2020dive,
    title={Dive into Deep Learning},
    author={Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola},
    note={\url{https://d2l.ai}},
    year={2020}
}

I collected all code for Transformer from d2l.ai in one notebook here:   
https://github.com/Dmitri9149/PyTorch-Transformer-from-scratch/blob/main/D2l_PyTorch_Transformer_from_Scratch_v0.ipynb   
(this repository).  
It is a 'reference point' to test changes in architecture. 

If I still use the d2l.ai code, I mark it as "\### from d2l.ai" or as "\### modified code from d2l.ai"

The new architecture of the Transformer I am testing now: 

1. MultiHeadAttention is rewritten in the way where every vector representation for a token is considered as "undivisible unit" , which better correspond to a notion of a vector. In the original impementation, the 'token vector' is split to #heads (number of heads) parts , every part is supplied to Linear transfromation, the results are concatenated. (Linear transformation is same for all parts). 

- In new architecture every 'vector-token' is supplied to Liner transformation (same for all 'vector-tokens') and the results are : 
- concatenated (see ...... as example) 
- TODO  -> are summed and notmilized   

2. The math analysis may suggest the Linear transformation for 'keys' is redundant: two matrix_A\*matrix_B may be reduced to just one learnable matrix_C. The Linear transformation for 'keys' is eliminated, it decrease the number of learning parameters. 
3. The PositionalEncoding module is new. The positions are encoded as (if num_steps is a number of tokens in a sentence): 
 1. linear function from a token position x in sequence of length num_steps, for example 
 - 0 ->  1.
 - (num_steps -1 ) -> -1 
 - linear change from 1. to -1. for all intermediate values of x
 2. as sequece of 0 and 1 with length num_steps (with some normalization)
 3. TODO : to use 'one hot encoding' for num_steps positions. 

 The work is in progress, the code is 'development code'. The are a lot of intermediate tests in code, comments etc ... 

 In the new architecture development the intuition is invaluable. The only way to get intuition is experimentation. 

The 'big' questions I have in mind with the (and similar) projects are: 
- 'Is is possible to do the machine language translation without a back propagation ?'
- 'How to do the machine language translation without a back propagation ?' 
- 'What is architecture in this case ?' 

In my opinion the answer is positive: yes, it is possible. There is to be an architecture which combine the Transformer ideas, Encoding/Decoding architecture in general,  and Bayesian , Support Vector Machine 'direct learning style'.

