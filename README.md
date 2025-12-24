# Awesome-LLMs-for-Clustering
Awesome-LLMs-for-Clustering is a curated collection of state-of-the-art methods, influential papers, open-source codes, and benchmark datasets related to applying large language models for clustering

## A Taxonomy of LLMs for clustering

- LLM as Representor
  - Direct Embedding
  - Representation Augmentation
    - Instance-level
    - Cluster-level
  - Multi-view Representation Learning
- LLM as Reasoner
  - Direct Assignment
  - Relative Reasoning
- LLM as Optimizer
  - Clustering Adaptation
  - Structure Constraint
  - Attention Optimization

## Resources

> Latest Updates: Dec 2025
### Overall Framework
![framework]()
  
| **Year** | **Method**             | **Title**                                                                                                                                   | **Paper**                                                                                                                                                                                                                                                                            | **Link**                                                                          |
| -------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| 2025     | **EHR-DeBERTa**        | **Identifying clusters of people with Multiple Long-Term Conditions using Large Language Models: a population-based study**                 | [link](https://www.nature.com/articles/s41746-025-01806-9.pdf)                                                                                                                                                                                                                       | [link](https://github.com/microsoft/DeBERTa)                                      |
| 2025     | **Petukhova et al.**   | **Text clustering with large language model embeddings**                                                                                    | [link](https://www.sciencedirect.com/science/article/pii/S2666307424000482)                                                                                                                                                                                                          | -                                                                                 |
| 2024     | **Zhao et al.**        | **Leveraging Large Language Models and Fuzzy Clustering for EEG Report Analysis**                                                           | [link](https://ieeexplore.ieee.org/abstract/document/10784894/)                                                                                                                                                                                                                      | -                                                                                 |
| 2024     | **Keraghuel et al.**   | **Beyond words: a comparative analysis of LLM embeddings for effective clustering**                                                         | [link](https://hal.science/hal-04488175v1/file/ida2024_LLM_paper.pdf)                                                                                                                                                                                                                | -                                                                                 |
| 2024     | **ERASMO**             | **ERASMO: Leveraging Large Language Models for Enhanced Clustering Segmentation**                                                           | [link](https://arxiv.org/pdf/2410.03738)                                                                                                                                                                                                                                             | [link](https://github.com/fsant0s/ERASMO)                                         |
| 2024     | **TCBPMA**             | **Text clustering based on pre-trained models and autoencoders**                                                                            | [link](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2023.1334436/full)                                                                                                                                                                     | -                                                                                 |
| 2025     | **Al-Saiidi et al.**   | **Privacy preservation embedding-based clustering for population stratification using large language models**                               | [link](https://ieeexplore.ieee.org/abstract/document/11081619)                                                                                                                                                                                                                       | -                                                                                 |
| 2024     | **Viswanathan et al.** | **Large language models enable few-shot clustering**                                                                                        | [link](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00648/120476)                                                                                                                                                                                                          | [link](https://github.com/viswavi/few-shot-clustering)                            |
| 2024     | **TAPE**               | **Harnessing explanations: Llm-to-lm interpreter for enhanced text-attributed graph representation learning**                               | [link](https://arxiv.org/pdf/2305.19523)                                                                                                                                                                                                                                             | [link](https://github.com/XiaoxinHe/TAPE)                                         |
| 2025     | **TextLens**           | TextLens: large language models powered visual analytics enhancing text clustering                                                          | [link](https://www.researchgate.net/profile/Yu-Dong-18/publication/388876076_TextLens_large_language_models-powered_visual_analytics_enhancing_text_clustering/links/67b69972207c0c20fa8e92d2/TextLens-large-language-models-powered-visual-analytics-enhancing-text-clustering.pdf) | -                                                                                 |
| 2025     | **LOGIN**              | **LOGIN: A Large Language Model Consulted Graph Neural Network Training Framework**                                                         | [link](https://dl.acm.org/doi/epdf/10.1145/3701551.3703488)                                                                                                                                                                                                                          | [link](https://github.com/QiaoYRan/LOGIN)                                         |
| 2025     | **Cafellm**            | **Cafellm: Context-aware fine-grained semantic clustering using large language models**                                                    | [link](https://arxiv.org/abs/2405.00988)                                                                                                                                                                                                                                             | [link](https://github.com/amazon-science/context-aware-llm-clustering)            |
| 2022     | **SimPTC**             | **Beyond prompting: Making pre-trained language models better zero-shot learners by clustering representations**                            | [link](https://aclanthology.org/2022.emnlp-main.587.pdf)                                                                                                                                                                                                                             | [link](https://github.com/fywalter/simptc)                                        |
| 2024     | **CSAI**               | **Large language model enhanced clustering for news event detection**                                                                       | [link](https://arxiv.org/pdf/2406.10552)                                                                                                                                                                                                                                             | -                                                                                 |
| 2024     | **Pattnaik et al.**    | **Improving hierarchical text clustering with llm-guided multi-view cluster representation**                                                | [link](https://aclanthology.org/2024.emnlp-industry.54.pdf)                                                                                                                                                                                                                          | [link](https://github.com/Observeai-Research/hierarchical-clustering-data-corpus) |
| 2025     | **HERCULES**           | **HERCULES: Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization**                                      | [link](https://arxiv.org/pdf/2506.19992)                                                                                                                                                                                                                                             | [link](https://github.com/bandeerun/pyhercules)                                   |
| 2025     | **AIME**               | **Improving Clustering Explainability and Automated Cluster Naming with Approximate Inverse Model Explanations and Large Language Models ** | [link](https://ieeexplore.ieee.org/abstract/document/10981499)                                                                                                                                                                                                                       | [link](https://github.com/ntakafumi/aime)                                         |
| 2024     | **TAC**                | **Image Clustering with External Guidance**                                                                                                 | [link](https://arxiv.org/pdf/2310.11989)                                                                                                                                                                                                                                             | [link](https://github.com/XLearning-SCU/2024-ICML-TAC)                            |
| 2024     | **Multi-Map**          | **Multi-modal proxy learning towards personalized visual multiple clustering**                                                              | [link](https://openaccess.thecvf.com/content/CVPR2024/papers/Yao_Multi-Modal_Proxy_Learning_Towards_Personalized_Visual_Multiple_Clustering_CVPR_2024_paper.pdf)                                                                                                                     | [link](https://github.com/Alexander-Yao/Multi-MaP)                                |
| 2024     | **Multi-Sub**          | **Customized multiple clustering via multi-modal subspace proxy learning**                                                                  | [link](https://proceedings.neurips.cc/paper_files/paper/2024/file/96b8167534ef3cc30c230bbeb55a524d-Paper-Conference.pdf)                                                                                                                                                             | [link](https://github.com/Alexander-Yao/Multi-Sub)                                |
| 2025     | **ESMC**               | **ESMC: MLLM-Based Embedding Selection for Explainable Multiple Clustering**                                                                | [link](https://arxiv.org/pdf/2512.00725)                                                                                                                                                                                                                                             | [link](https://github.com/JCSTARS/Embedding-Selective-Multiple-Clustering)        |
|  2025        |      **LLM-DAMVC**                  |      **LLM-DAMVC: A Large Language Model Assisted Dynamic Agent for Multi-View Clustering**                                                                                                                                       |     [link](https://openreview.net/pdf?id=xgiMK8FtSI)                                                                                                                                                                                                                                                                                 |   -                                                                                |
|  2025        |      **VISTA**                  |        **VISTA: A Multi-View, Hierarchical, and Interpretable Framework for Robust Topic Modelling**                                                                                                                                     |    [link](https://www.mdpi.com/2504-4990/7/4/162)                                                                                                                                                                                                                                                                                  | [link](https://github.com/domjanbaric/vista)                                                                                  |
|   2025       |      **LiSA**                  |      **LLM-Guided Semantic-Aware Clustering for Topic Modeling**                                                                                                                                       |   [link](https://aclanthology.org/2025.acl-long.902.pdf)                                                                                                                                                                                                                                                                                   |  [link](https://github.com/ljh986/LiSA)                                                                                 |                                                                             |


## Datasets
Please stay tuned for more updates!

## Related Repos

### Clustering
- [Awesome-Deep-Graph-Clustering](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering)
- [PyDGC](https://github.com/Marigoldwu/PyDGC)
- [Awesome-Deep-Clustering](https://github.com/zhoushengisnoob/DeepClustering)
- [Awesome-Deep-Multi-view-Clustering](https://github.com/zhangyuanyang21/Awesome-Deep-Multi-view-Clustering)

### LLMs
- [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)
- [Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference)
- [Awesome-LLM-Reasoning](https://github.com/atfortes/Awesome-LLM-Reasoning)
- [Awesome-LLM-Resources](https://github.com/WangRongsheng/awesome-LLM-resources)
- [Awesome-Domain-LLM](https://github.com/luban-agi/Awesome-Domain-LLM)
- [Awesome-Graph-LLM](https://github.com/XiaoxinHe/Awesome-Graph-LLM)
- [Awesome-LLMs-Datasets](https://github.com/lmmlzn/Awesome-LLMs-Datasets)
- [Awesome-LLM-Interpretability](https://github.com/JShollaj/awesome-llm-interpretability)

## Pull Requests Templates

We will track the development of LLM-for-Clustering and update the list frequently. We also welcome your PR. If you submit a PR, we recommend that you follow the template below. Sincere thanks to you!

- If a list of papers from a new year, please add the following code (【】 are required contents).

- If there is no open source code, please delete <img src='./assets/code.png' /> <a href='【Code URL】' target='_blank'>Code</a>. It is recommended to add open source code, and prioritize items with code before items without code.

```html
<details open> <!--2025-->
  <summary><b>&nbsp;【YEAR】 (【Numbers】)</b></summary>
  <ul>
    <li>
      <b>【Abbreviation】</b><i>【Article Name】. <b>【Publication'Year】</b>.</i>
      <img src='./assets/paper.png' /> <a href='【Article URL】' target='_blank'>Article</a>
      <img src='./assets/code.png' /> <a href='【Code URL】' target='_blank'>Code</a>
    </li>
  </ul>
</details>
```

## Citation
If you find this repository useful in your research, please consider citing:

```bibtex
Please stay tuned for more updates!
```

## Contributors
Thanks to the following contributors for their contributions to this repository:

<a href="https://github.com/fuyw-aisw" target="_blank"><img src="https://avatars.githubusercontent.com/u/118046924?v=4" alt="yueliu1999" width="96" height="96"/></a> 
<a href="https://github.com/Marigoldwu" target="_blank"><img src="https://avatars.githubusercontent.com/u/75920051?v=4" alt="Marigoldwu" width="96" height="96"/></a> 

<p align="right">(<a href="#top">back to top</a>)</p>
