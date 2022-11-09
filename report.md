# Report
Much of the information I discuss here is covered in more detail in my attached jupyter notebook.
Please feel free to refer to that, or email/slack if additional information is required.


## 1: Understand and distinguish concepts:

*   **Sparsification** is the process of decreasing the number of edges in the network while maintaining some topological properties of the network. Edges are typically "decreased" by setting their weights equal to zero
*   **Pruning** is the process of removing edges from a network. It typically refers to setting a particular weight to 0 and freezing it for training. This is sometimes referred to as "applying a mask" to the weights
*   **Quantization** is the mapping of continous values (or otherwise very large set of values) to set of discrete values
*   **Distillation** is the process of transferring "knowledge" from a larger model to a smaller model. Logits, weights, or activations can be used as the source of "knowlege"
*   **MoEfication** (I was only able to find one paper referencing this: https://arxiv.org/pdf/2110.01786.pdf ) the transforming of a model to its MoE (Mixture of Experts) version via partitioning the internal network

## 2. Choose your models

*   Encoder-only: DeBERTa v2: 1.5B params (https://arxiv.org/abs/2006.03654)

*   Encoder-Decoder: (3B params) FLAN-T5 (https://huggingface.co/docs/transformers/model_doc/flan-t5)

*   Decoder-only: GPT-2: 1.5B params (https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

I refer to DeBERTa v2 as "DeBERTa" for shorthand henceforth.

## 3. Devise approaches to assess sparsity structure in your choice of models and answer these questiosn: what fraction of parameters >> 0? overall? by layer? how does this vary by layer?

Note: I am reporting the fraction of parameters, not percents.

###	DeBERTa:
using the >> 0 cutoff of 0.2
Total fraction of parameters >> 0? (overall) = 0.00019522949501320168
Total fraction of parameters >> 0 (by layer), and how it varies by layer:
![This is an image](https://github.com/madisonPickering/llm-sparsification-release-madisonp/blob/master/src/NonzeroParams_deberta%20v2.png)


### Flan-T5
using the >> 0 cutoff of 0.2
Total fraction of parameters >> 0? (overall) = 0.45902937682966827
Total fraction of parameters >> 0 (by layer), and how it varies by layer:
![This is an image](https://github.com/madisonPickering/llm-sparsification-release-madisonp/blob/master/src/NonzeroParams_flan-t5.png)

### GPT-2
using the >> 0 cutoff of 0.2
Total fraction of parameters >> 0? (overall) = 0.12839315783068259
Total fraction of parameters >> 0 (by layer), and how it varies by layer:
![This is an image](https://github.com/madisonPickering/llm-sparsification-release-madisonp/blob/master/src/NonzeroParams_gpt-2.png)

## 4. Produce sparsified versions of your models at 10%, 50%, 90%, 95%, 99%, by either coding your methods or using existing tools provided below Explain the nature of your methods, regardless of whether you code it yourselves.
I used the existing tool of pytorch prune, to do global pruning.
**Explanation of my methods:**
I will use the existing tool of Pytorch Prune. I will specifically use global pruning, which, given the layers to prune, will prune weights by "removing" (setting = 0) the lowest (in abs magnitude) x% of weights over the model globally, where x is specified by the programmer. Since weights are pruned in a global manner, there may be more or less than x% of weights pruned per layer, however, x% of weights will be pruned over the entire ("global") model.

## 5.Find 2 common benchmarks used by your models, by reviewing their publications.
Set them up and obtain baseline results of original models.
Compare performance of your sparsified versions with the baselines. Include plots and explanations


**Common benchmarks:** I chose to use perplexity on wiki-2 and wiki-103 as my two benchmarks. A more detailed explanation of my justification for this is in my attached notebook: 
https://github.com/madisonPickering/llm-sparsification-release-madisonp/blob/master/src/asst4_sparsification.ipynb

**Explanation:** For all graphs, the baseline model is at x=0 (aka, 0% pruned)--the x axis is percent pruned. The y axis is perplexity.
I noticed, in general, that perplexity increased (in a nonlinear/noisy manner) as the model was pruned more. I also noticed that pruning at 10% often had a negligible impact on
model performance, while pruning at the 90, 95, 99% typically had very noticable effects. (Probably because so few weights remained at that point.)
to be DeBERTa, which had a fairly high perplexity regardless of if it was pruned or not. When I looked into the manner further, I realised that
huggingface (or, more specifically, microsoft) had not officially released a version of DeBERTa with a properly instantiallized LM head. I similarly did not find reputable third
party sources (i.e., third parties which included any documentation with their model release) which had extended microsoft's base DeBERTa model
by properly adding weights to the model's LM head. I leave this issue open to future work.

### wiki-2:
![This is an image](https://github.com/madisonPickering/llm-sparsification-release-madisonp/blob/master/src/debert_perplex_2.png)
![This is an image](https://github.com/madisonPickering/llm-sparsification-release-madisonp/blob/master/src/gpt2_perplex_2.png)
![This is an image](https://github.com/madisonPickering/llm-sparsification-release-madisonp/blob/master/src/t5_perplex_2.png)


### wiki-103:
![This is an image](https://github.com/madisonPickering/llm-sparsification-release-madisonp/blob/master/src/debert_perplex_103.png)
![This is an image](https://github.com/madisonPickering/llm-sparsification-release-madisonp/blob/master/src/gpt2_perplex_103.png)
![This is an image](https://github.com/madisonPickering/llm-sparsification-release-madisonp/blob/master/src/t5_perplex_103.png)


## 6. Compare size of models and runtime for sparsified models. Include plots and explanations

**Explanation**: In general, I observed that inference required less time the more the model is pruned. I also noticed that time to execute was fairly noisy.
As before, the x axis is % pruned. The y axis is run time in seconds.

### wiki-2:
![This is an image](https://github.com/madisonPickering/llm-sparsification-release-madisonp/blob/master/src/debert_time_2.png)
![This is an image](https://github.com/madisonPickering/llm-sparsification-release-madisonp/blob/master/src/gpt2_time_2.png)
![This is an image](https://github.com/madisonPickering/llm-sparsification-release-madisonp/blob/master/src/t5_time_2.png)


### wiki-103:
![This is an image](https://github.com/madisonPickering/llm-sparsification-release-madisonp/blob/master/src/debert_time_103.png)
![This is an image](https://github.com/madisonPickering/llm-sparsification-release-madisonp/blob/master/src/gpt2_time_103.png)
![This is an image](https://github.com/madisonPickering/llm-sparsification-release-madisonp/blob/master/src/t5_time_103.png)

## 7.Explain the challenges of sparsification on LLMs.
I found the main challenges to be:

*	1. The computational cost (RAM) of (global) sparsification
*	2. Performance does not degrade linearly as sparsification occurs, making it difficult to determine exactly to what degree one should sparsify a model
*	3. Given multiple weights which are candidates for pruning, it is not clear which weights should be pruned over others. That is, it will become clear after benchmarking/other experimentation, however benchmarking, experimentation is time consuming.
*	4. Computational time for inference does not decrease linearly as sparsification occurs. I found in my experiments that there was a lot of "noise" in my observed measurements. This similarly makes it harder to gauge the exact benefits of sparsification.


