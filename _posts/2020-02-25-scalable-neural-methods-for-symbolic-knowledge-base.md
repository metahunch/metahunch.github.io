---

layout: post
title: Scalable Neural Methods for Reasoning with a Symbolic Knowledge Base
date: 2020-02-25
author: Changlong Yu
comments: true

---



- toc
{:toc}
## TLDR

This paper introduces a scalable and efficient way to incorporate the whole semantics of KB into neural models so as to conduct multi-hop inferences over large-scale KBs.  Specially a symbolic KB is represented by three sparse matrices and inferences are conducted by distributed matrix manipulation in several GPUs. Neuralized KB reasoning is handled by weighted relation vectors learned from a neural module.  Please refer to the following parts for details. 



## Reasoning Over the Whole KB

### Motivation

Injecting external knowledge from existing knowledge bases (graphs) into neural models is a hot but non-trivial problem, which benifits many downstream tasks *i.e.,* KBQA.   

Previous works simply adopt entity vectors or relation vectors as features, derived from knowledge embedding algorithms such as TransE,  ConvE and DistMult and do not support multi-step inference effectively. Reasoning over KBs mostly focuses on local graphs considering the unrealistic encoding the whole KB (one real KB has 13 million entities and 44 million facts). This paper propopses a novel way to fully encode the symbolic KB so as to fit to GPU memory and effective inference methos to incorporate into neural models. 

### Preliminary and Notation

> The original paper is very hard to follow in terms of  notations and organizations. I briefly summarize the key ideas and experimental insights.

***Basic definition:*** Normally in a KB, there are a set of entities $x \in X$ and a set of relations $r \in R$.  Denote $N_e$ be the number of entities and $N_r$ be the number of relations.  A relation $r$ could be encoded as a matrix $M_r \in \mathcal{R}^{N_e \times N_e}$ [^1], where $M_r[i,j] \in {\lbrace 0,1 \rbrace}$ means that whether a pair of entities $(x_i, x_j)$ has a relation $r$.   To neuralize the representations,  a weighted entity set $X$ is encode as a vector $\mathbf{x} \in \mathcal{R}^{N_e}$, which $\mathbf{x}_{i}$ is the weight of $x_i$ in $X$. Similiarly a relation set $R$ is encoded as  $\mathbf{r} \in \mathcal{R}^{N_r}$. 
***Relation following operation:***  Multi-hop reasoning over knowledge graphs actually is equivalent to computing the neighbors of multiple steps originated from an entity.  The authors define the operation of *R-neighbors* for an entity set $X$,  i.e., find all the first-order neighbors of the enity in the $X$ under all the realtions in the $R$.  
Let $M_{R}$ be a weighted msixture of relation matrices of all relations, 

$$\mathbf{M}_R \equiv \sum_{k=1}^{N_R} \mathbf{r}[k] \cdot M_{r_k}$$

Hence the approximate *R-neighbors* with differentiable operation could be reformulated as: 

$$follow(\mathbf{x}, \mathbf{r}) \equiv \mathbf{x}\mathbf{M}_R = \mathbf{x}(\sum_{k=1}^{N_R} \mathbf{r}[k] \cdot M_{r_k}) \tag{1}$$ 

Intuitively matrix multiplication is to find the approximations of adjacent nodes assoicated with all types of  relation edges.  We could easily generalize to $T-$hop reasoning:

$$\mathbf{x}^{t} = follow(\mathbf{x}^{t-1}, \mathbf{r}^t) \tag{2}$$ 

### Scalable Representations

As we can see from Eq.$1$ , sparse-matrix operation could not be extended into minibatches, which is not efficent and scalable for large-scale KBs.  The most ingenious part is to decompose $\mathbf{x}\mathbf{M}_R$ sparse operation into the following form:
$$follow(x,r) = (\mathbf{x} \mathbf{M}^{T}_{subj} \odot \mathbf{r} \mathbf{M}^{T}_{rel}) \mathbf{M}_{obj} \tag{3}$$  
The three sparse matrices ${\mathbf{M}_{subj}} \in {\lbrace 0,1 \rbrace}^{N_t \times N_e}$ are derived from all the tuples of KBs for example the $\mathcal{l}$-th tuple  $(i_l,j_l,k_l)$ [^2]  from KB assertion $r_k(x_i,x_j)$, ${\mathbf{M}_{subj}}[\mathcal{l}, i_l] =1$,  ${\mathbf{M}_{rel}}[\mathcal{l}, k_l] =1$  and ${\mathbf{M}_{obj}}[\mathcal{l}, j_l] =1$. 

> $\mathbf{x} {\mathbf{M}^{T}_{subj}}$ are the triples with an entity in  $\mathbf{x}$ as their subject, $\mathbf{r} {\mathbf{M}^{T}_{rel}}$ are the triples with a relation in $\mathbf{r}$. The final multiplication finds the object entities of the tuples in the interaction. 
In such way,  Eq.$3$ could be easily extended to mini-batches with batch size of $b$, i.e,  $\mathbf{X}\in \mathcal{R}^{b \times N_e}$, $\mathbf{R}\in \mathcal{R}^{b \times N_r}$  
$$follow(\mathbf{X},\mathbf{R}) = (\mathbf{X}\mathbf{M}^{T}_{subj} \odot \mathbf{R}\mathbf{M}^{T}_{rel}) \mathbf{M}_{obj} \tag{4} $$
By reformulating the *follow* operation in more efficient ways,  it is eaiser to parallelly run a batch of data and distribute three large sparse matrices into serveal GPUs with limited memory.   
### Incorporating to Neural Models
- KBQA:  The training data for this task is $(q,A)$ , where $q$ is natural language question and $A$ is a set of KB entities.  Using the aforementioned KB for reasoning means neural models could predict the relations from the initial entity set $\mathbf{x^0}$ to the desired answers $A$ along the reasoning process.

  $$\mathbf{r}^{t} = f^{t}(q) \ for \ t \in \{1, 2, ... ,T\} \tag{5}$$

  For each step $t$,  neural model $f^{t}(q)$ encodes the question $q$ into the weighted relation vector $\mathbf{r}^{t}$, where each element in $\mathbf{r}^{t}$ is the weight of corresponding relation in $R$.  Then apply Eq.$2$ to compute the inferenced set $\mathbf{x}^T$ and cross-entropy loss is used between $\mathbf{x}^{T}$ and answer $A$ for optimization. 

  > Comments:  We could think about the inference process in the logical way. With the current entity nodes $\mathbf{x^{t-1}}$ at hand,  neural model $f^{t}(q)$ and $follow(\mathbf{x}^{t-1}, f^{t}(q))$ decide where and how to traverse  over the knowledge graph.     

- Knowledge Base Completion:  As an important application of KBs, KBC aims to predict tail enities given a head entity $\mathbf{x}$ and a relation $r$ . For example (Obama, isPresidentOf,  $?$).  Actually there exist multiple paths from the head entity node to the desired nodes and the length of those paths (the steps of inference) varies.  We should predefine the hyperparameters $N$ (the number of inference chains) and $T$ (the maximum steps of inference).  Similiar to KBQA, the predicted tail entities $\hat{\mathbf{a}}$ would be the *softmax* of all inference chains. 

  $$ \hat{\mathbf{a}} = softmax(\sum^{N}_{i=1} \mathbf{x}^{T}_{i}) \tag{6}$$

  

## Summary 

### Subjective Comments

1. From the perspective of my understanding (may be wrong),  this paper solves the problem that multi-hop inferences (matrix manipulation) over symbolic KBs represented by huge sparse matrix $\mathbf{M}_R$ could not fit into limited GPU memory. 
   - The authors refine the KB representations by decomposing KB tuples $(x_i, x_j, r_k)$ into three sparse matrices ${\mathbf{M}_{subj}}$.  Essentially it is similiar to  explicit **tensor decomposition** in Eq.$3$.
   - Batching input data and distributing matrix operations would allow for reasoning over ten-million scale of knowledge graphs. 
   - Could multi-hop inferences in Eq.$2$ be regarded as **soft transitive closure**？Starting from the initial nodes, check the reachability of targeted nodes in the knowledge graphs. 

2. In terms of neuralized KB reasoning,  this paper mainly use the learned weighted relation vector $\mathbf{r}$ to control the weight of each realtion along reasoning steps. The core inference ability is determined by the weight distribution of relations on basis of current entity set $\mathbf{x}^{t}$. 

   - Could we interpret the relation vector $\mathbf{r}$ as the a type of **attention mechanism**?  
   - If we consider both the query $q$ and current entity set $\mathbf{x}^{t-1}$ at step $t$ to parameterize neural models, i.e.,   $\mathbf{r}^{t} = f^{t}(q,\mathbf{x}^{t-1})$ ，is this necessary? 

3. What is the real advantage of reasoning over the whole symbolic KBs rather than local sub-graphs?  This paper mentions that their methods are fully differentiable that gradients could be propogated to original KB inputs and keep the complete semantics. 

   - It is ture that the end-to-end reasoning avoids the integration of **retrieval then neural process**. 

   - The insightful conclusion is that the proposed method is good at **long-distance reasoning**. 

   - > The competitive performance of using fewer parameters (only $T$ neural networks) is directly realted to the fact that our model directly uses inference on the existing symbolic KB in its model, rather than having to learn embeddings that **approximate this inference**.

4. The proposed methods are supposed to somehow provide the **explainability of model inferences**. The authors fail to give any analysis about reasoning paths. We are interested in whether the inferences could find the optimal path or any case study as [ICLR2018]. 

5. Another concern is the **incompleteness of KB**. The logical inference chains in this work would somehow relieve such issue but the generalisation for logical reasoning remains unclear (maybe think more). 

### Take-aways

- Matrix (tensor) decomposition is an effective way for large-scale distributed computations though it might be a little hard to implement. 
- Overall the reasoning framework is symbolic inference along the graph paths and neural models are adopted to learn the edge weights (different types of edges for different relations).  
- One possible extension of this work is considering the **surface semantics of entities** besides the structure information of graphs, i.e., the node features. 

## Further Readings

- [Full Paper](https://openreview.net/forum?id=BJlguT4YPr)
- [[AAAI2018] Variational Reasoning for Question Answering with Knowledge Graph](https://arxiv.org/abs/1709.04071) 
- [[ICLR2018] Reasoning over Paths in Knowledge Bases using Reinforcement Learning](https://openreview.net/pdf?id=Syg-YfWCW) 
- [[ICLR2020] Efficient Probabilistic Logic Reasoning with Graph Neural Network](https://arxiv.org/pdf/2001.11850.pdf)
- <a href="{{ page.url }}/demo-comment">More Comments</a>

[^1]: We actually need sparse matrix to store matrix $M_r$ for each relation since the real KBs have ten million entities.
[^2]: Denote $N_t$ be the number of all the tuples in the KB. 

