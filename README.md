<!-- <style>
H1{color:Blue !important;}
H2{color:DarkOrange !important;}
p{color:Black !important;}
</style> -->

CS 7641 Team 1

Shreyas Verma, Saksham Arora, Mehul Soni, Samaksh Gulati, Sindhu Raghuram Panyam

## Introduction
Our problem statement and methodology is  primarily driven by the advances in NLP that have created significant impact in healthcare for both patients and physicians.
With these guidelines, we narrowed down to a problem statement related to reducing the burden of conducting the United States Medical Licensing Examination taken by medical professionals each year. 

One phase of this Examination, namely Step 2 Clinical Skills involves test-takers interacting with Standardized Patients (people trained to portray specific clinical cases) and writing  patient notes. Trained physician raters later scored patient notes with rubrics that outlined each case’s important concepts (referred to as features).
Many techniques have been employed to tackle this problem statement. One such paper<sup>[6]</sup> tackles the task of domain specific passage matching by using a rule-based self-supervision objective for training Transformer language models.


Upon extensive literature review, we realize that the following methodologies are interesting avenues that can be explored for our task :
1. <b>Effect of domain specific pretraining - </b>Large Language models, trained on general English corpus, might lead to an under-representation of domain-specific term embeddings in the corpus. Research in tweaking the pretraining phase focuses on task-adaptive pretraining of models like BERT using static word embeddings (fastText) in conjunction with different regularization techniques<sup>[3]</sup>, or framing subtasks<sup>[4]</sup> to explicitly learn the rare embeddings. Research in the fine-tuning phase has worked towards explaining rare words with paraphrases on the basis of prompt-based tuning techniques<sup>[5]</sup>. The former, albeit computationally expensive, is seemingly better - encouraging its use in representations of rare biomedical terminologies. 

2. <b>Instruction based fine tuning - </b>Recent works in prompting techniques<sup>[1][2]</sup> have explored multi-task instruction-based fine tuning. They demonstrate that instruction tuned models have a great ability to generalize and can perform arduous generative tasks with ease. Using that idea as an inspiration we propose instruction based fine-tuning. 

Recently introduced instruction--tuned models have the potential to significantly outperform task-specific language models (without instruction), but their effect has not yet been systematically studied in the healthcare domain. 

The dataset presents a corpus of 43,985 clinical patient notes (PNs) written by 35,156 examinees during the high-stakes USMLE® Step 2 Clinical Skills examination.

[
<img width="813" alt="Screen Shot 2022-10-07 at 12 33 48 AM" src="https://user-images.githubusercontent.com/65465058/194675093-ab31dfe7-240f-4d50-9e4b-aa249b7c9d61.png">
](url)

<!-- [
<img width="813" alt="Screen Shot 2022-10-07 at 12 33 48 AM" src="https://raw.githubusercontent.com/samaksh97/Instruction-Tuned-Clinical-Notes-Scoring/main/Pictures/label.png?token=GHSAT0AAAAAABZV3HB7XANMBBBMV6ELTK44Y2A6CZA">
](url)
 -->
## Problem Definition

The goal of our project is to develop an automated way of identifying the semantically relevant features within each patient note, documenting standardized patient interviews. 

## Methods
When working on a specific domain, it is ideal to pre-train the LLM on rare words first and then perform the domain-specific task. Here, we plan to first pre-train the instruction-tuned [Tk-Instruct LLM](https://huggingface.co/allenai/tk-instruct-3b-def-pos-neg)<sup>[2]</sup>  so that it learns the rare biomedical terms. We will then fine-tune it using Instructional prompts composed of a definition and a positive example. 
<!-- ![Tk-instruct](https://user-images.githubusercontent.com/65465058/194685081-51c1b248-27c9-4441-9b89-8eb3ee671fa3.png| width=100) -->

<img src="https://user-images.githubusercontent.com/65465058/194685081-51c1b248-27c9-4441-9b89-8eb3ee671fa3.png" width="500">
<!-- 
[
<img width="813" alt="Screen Shot 2022-10-07 at 12 33 48 AM" src="https://raw.githubusercontent.com/samaksh97/Instruction-Tuned-Clinical-Notes-Scoring/main/Pictures/Tk-instruct.png?token=GHSAT0AAAAAABZVW6OKZ45CSTR4OZ3LZBFWY2A5ULQ">
](url) -->

## Potential Results and Discussions
As our aim is to identify semantically similar phrases from the patient notes that match the medically termed features, we plan to use <b>micro-averaged F1 score</b> to evaluate the overlap of the predicted phrase spans with the label-spans.

## References
1. Mihir Parmar, Swaroop Mishra, Mirali Purohit, Man Luo, Murad Mohammad, and Chitta Baral. 2022. In-BoXBART: Get Instructions into Biomedical Multi-Task Learning. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 112–128, Seattle, United States. Association for Computational Linguistics.
2. Wang, Y., Mishra, S., Alipoormolabashi, P., Kordi, Y., Mirzaei, A., Arunkumar, A., Ashok, A., Dhanasekaran, A.S., Naik, A., Stap, D. and Pathak, E., 2022. Benchmarking generalization via in-context instructions on 1,600+ language tasks. arXiv preprint arXiv:2204.07705.
3. Kosuke Nishida, Kyosuke Nishida, and Sen Yoshida. 2021. Task-adaptive Pre-training of Language Models with Word Embedding Regularization. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 4546–4553, Online. Association for Computational Linguistics.
4. Schick, T., & Schütze, H. (2020). Rare Words: A Major Problem for Contextualized Embeddings and How to Fix it by Attentive Mimicking. Proceedings of the AAAI Conference on Artificial Intelligence, 34(05), 8766-8774. https://doi.org/10.1609/aaai.v34i05.6403
5. Wang H, Liu C, Xi N, Zhao S, Ju M, Zhang S, Zhang Z, Zheng Y, Qin B, Liu T. Prompt Combines Paraphrase: Teaching Pre-trained Models to Understand Rare Biomedical Words. arXiv preprint arXiv:2209.06453. 2022 Sep 14.
6. Grundmann, Paul, Sebastian Arnold, and Alexander Löser. "Self-supervised answer retrieval on clinical notes." arXiv preprint arXiv:2108.00775 (2021).

