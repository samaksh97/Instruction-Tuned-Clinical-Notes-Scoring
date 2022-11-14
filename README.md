<!-- <style>
H1{color:Blue !important;}
H2{color:DarkOrange !important;}
p{color:Black !important;}
</style> -->

CS 7641 Team 1

Mid Term Project
Shreyas Verma, Saksham Arora, Mehul Soni, Samaksh Gulati, Sindhu Raghuram Panyam

[Gantt Chart](https://docs.google.com/spreadsheets/d/127ndniyh15_XrMLIP_JnUK5plsJSzKy0/edit?usp=sharing&ouid=105287803295740692275&rtpof=true&sd=true)

[Proposal Video](https://drive.google.com/file/d/1Rn7oqu6s77t66SFFsS9U225rkypubJZz/view?usp=sharing)

## Introduction
Recently introduced instruction-paradigm empowers non-expert users to leverage NLP resources by defining a new task in natural language. Instruction-tuned models have the potential to significantly outperform task-specific language models (without instruction), but their effect has not yet been systematically studied in the healthcare domain. Also, Large Language models have been trained on a general English corpus and may not perform well on domain-specific domains which have a high proportion of rare words.

Our problem statement and methodology is  primarily driven by the advances in NLP that have created significant impact in healthcare for both patients and physicians.
With these guidelines, we narrowed down to a problem statement related to reducing the burden of conducting the United States Medical Licensing Examination taken by medical professionals each year. 

One phase of this Examination, namely Step 2 Clinical Skills involves test-takers interacting with Standardized Patients (people trained to portray specific clinical cases) and writing  patient notes. Trained physician raters later scored patient notes with rubrics that outlined each case’s important concepts (referred to as features).


[
<img width="813" alt="Screen Shot 2022-10-07 at 12 33 48 AM" src="https://user-images.githubusercontent.com/65465058/194675093-ab31dfe7-240f-4d50-9e4b-aa249b7c9d61.png">
](url)



Many techniques have been employed to tackle this problem statement. One such paper<sup>[6]</sup> tackles the task of domain specific passage matching by using a rule-based self-supervision objective for training Transformer language models. Upon extensive literature review, we realize that the following methodologies are interesting avenues that can be explored for our task :

1. <b>Effect of domain specific pretraining - </b>Large Language models, trained on general English corpus, might lead to an under-representation of domain-specific term embeddings in the corpus. Research in tweaking the pretraining phase focuses on task-adaptive pretraining of models like BERT using static word embeddings (fastText) in conjunction with different regularization techniques<sup>[3]</sup>, or framing subtasks<sup>[4]</sup> to explicitly learn the rare embeddings. Research in the fine-tuning phase has worked towards explaining rare words with paraphrases on the basis of prompt-based tuning techniques<sup>[5]</sup>. The former, albeit computationally expensive, is seemingly better - encouraging its use in representations of rare biomedical terminologies. 

2. <b>Instruction based fine tuning - </b>Recent works in prompting techniques<sup>[1][2]</sup> have explored multi-task instruction-based fine tuning. They demonstrate that instruction tuned models have a great ability to generalize and can perform arduous generative tasks with ease. Using that idea as an inspiration we propose instruction based fine-tuning. 

Recently introduced instruction--tuned models have the potential to significantly outperform task-specific language models (without instruction), but their effect has not yet been systematically studied in the healthcare domain. 

The dataset presents a corpus of 43,985 clinical patient notes (PNs) written by 35,156 examinees during the high-stakes USMLE® Step 2 Clinical Skills examination.



## Problem Definition

USMLE® Step 2 Clinical Skills examination is one of the most prestigious and rigorous medical licensure exams for medical professionals. The exam required test-takers to interact with Standardized Patients (people trained to portray specific clinical cases) and write a patient note. Trained physician raters later scored patient notes with rubrics that outlined each case’s important concepts (referred to as features). 

The goal of the project is to extract substrings from the patient notes that are semantically closer to the feature at hand using an instruction-based learning of a pre-trained language model. From a project point of view, our goal is to study the different settings of instructional learning tasks (Instruction, Instruction+1 static example, Instruction + randomly sampled examples from a static pool) and compare results to identify optimal performance for our problem statement.


## Data Definition

The database presents a corpus of 43,985 clinical patient notes (PNs) written by 35,156 examinees during the high-stakes USMLE® Step 2 Clinical Skills examination. 
In medical education, students are often assessed through encounters with standardised patients - people trained to portray simulated scenarios called clinical cases. For each such encounter, the student is expected to perform a history and physical examination, determine differential diagnoses, and then document their findings in a PN.  

During this exam, each test taker sees a Standardised Patient, a person trained to portray a clinical case. After interacting with the patient, the test taker documents the relevant facts of the encounter in a patient note. Each patient note is scored by a trained physician who looks for the presence of certain key concepts or features relevant to the case as described in a rubric.

Clinical Case: The scenario (e.g., symptoms, complaints, concerns) the Standardised Patient presents to the test taker (medical student, resident or physician). Ten clinical cases are represented in this dataset.
Patient Note: Text detailing important information related by the patient during the encounter (physical exam and interview).
Feature: A clinically relevant concept. A rubric describes the key concepts relevant to each case.


## Data Processing
We used the following data preprocessing techniques
1. Created Input text using Patient Note and Feature
2. Appended an Example to pass in  along with an instruction (See Below)  
3. Added “No text found” for instances where feature was not found in Patient Note
4. We randomly split the dataset (14,300 Notes) into train and test splits as follows:
    80% - Training Set (11,400 Notes)
    20% - Testing Set (2,860 Notes)



<img src="https://user-images.githubusercontent.com/65465058/201565889-20929121-f57c-4774-8b36-7fbadf7ef5bf.png" width="500">

## Methods

When working on a specific domain, it is ideal to pre-train the LLM on rare words first and then perform the domain-specific task. Here, we plan to first pre-train the instruction-tuned [Tk-Instruct LLM](https://huggingface.co/allenai/tk-instruct-3b-def-pos-neg)<sup>[2]</sup>  so that it learns the rare biomedical terms. We will then fine-tune it using Instructional prompts composed of a definition and a positive example. 


<img src="https://user-images.githubusercontent.com/65465058/194685081-51c1b248-27c9-4441-9b89-8eb3ee671fa3.png" width="500">


When working on a specific domain, it is ideal to pre-train the LLM on rare words first and then perform the domain-specific task. Here, we plan to first pre-train the instruction-tuned T5 LLM so that it learns the rare biomedical terms. We will then fine-tune it using Instructional prompts composed of a definition and a positive example. 

At the outset, we began scoping different LLM models. However, while exploring different free cloud platform capabilities we realised that pretraining would not be feasible given the limitations of compute power. 

We have thus trained the T5 model with ~13000 instances. The parameters we have used in the model are as follows:


<img src="https://user-images.githubusercontent.com/65465058/201564479-de72c6e0-15b1-4227-a0fc-c4726d074fde.png" width="500">



As stated in the data processing section, we have currently used a static example for all training inputs. In addition to the outputs of the model, we can extract the weights of the final layer that have been updated while training which can be reused.

## Evaluation:
As our task can be framed as a span detection task - where we extract a span of words from the patient note as output - we decided to use 2 primary metrics for our task.The Exact Match metric measures the percentage of predictions that match any one of the ground truth answers exactly. The F1 score metric is a loser metric measures the average overlap between the prediction and ground truth answer. These scores are calculated on every prompt-response pair. Overall EM and F1 scores are computed for a model by averaging over the individual example scores. 

**Exact Match**:
For each prompt-response pair, the value of Exact Match metric will be 1 when the characters of the model’s prediction and that of the ground truth match exactly. This metric is comparatively more conservative; if a match is not found, the value is assigned to be 0. A difference in one character would result in a score of 0. 

**F1 Score**:
This metric is utilized when equal weightage is assigned to precision and recall. In this case, an attempt is made to find a match between the words in the prediction and the ground truth. 

We will be using a modified version of F1-score. This is because in our case we value recall more than precision. That is, it is important that in the output that the model gives us, the relevant feature text is present. However, we would not want to penalize the model as much for providing us with text in addition to the exact relevant phrase. Thus we will be using a Weighted F1-Score with greater emphasis on Recall rather than precision.

Having examined the resulting metrics, we get -

Tk-instruct small: 
**Avg f1-score - 67.5%**
**Avg Exact match score - 51.8%**

From these results, it can be observed that a respectable F1 score has been obtained. This implies that our predictions are quite close to some of the True Answers. The results also suggest that there is an exact match found between the predictions and the ground truth to a certain extent. However, we can explore the potential for enhancing these scores by increasing the complexity of the model (using Tk-instruct base).

## Potential Results and Discussions:
As our aim is to identify semantically similar phrases from the patient notes that match the medically termed features, we plan to use a micro-averaged F1 score to evaluate the overlap of the predicted phrase spans with the span of the labels.

## Plans Ahead 
As next steps, we plan to compare our current model performance with GPT-3 and evaluate scope of improvement. One possible improvement could be in the example that we have provided. We will be testing out adding multiple/dynamic examples instead of the same static example for all instances. We might also use other models containing more parameters in case compute constraints allow us to do so. Finally, we plan to explore the use of topic modelling and patient notes segmentation to address this problem.

## Updates and Changes to Scope:
As part of our evaluation of compute resources, we would not be going ahead with pre-training as previously planned.

As stated in the data processing section, we have currently used a static example for all training inputs. In addition to the outputs of the model, we can extract the weights of the final layer that have been updated while training which can be reused.

## Potential Results and Discussions
As our aim is to identify semantically similar phrases from the patient notes that match the medically termed features, we plan to use <b>micro-averaged F1 score</b> to evaluate the overlap of the predicted phrase spans with the label-spans.

## References
1. Mihir Parmar, Swaroop Mishra, Mirali Purohit, Man Luo, Murad Mohammad, and Chitta Baral. 2022. In-BoXBART: Get Instructions into Biomedical Multi-Task Learning. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 112–128, Seattle, United States. Association for Computational Linguistics.
2. Wang, Y., Mishra, S., Alipoormolabashi, P., Kordi, Y., Mirzaei, A., Arunkumar, A., Ashok, A., Dhanasekaran, A.S., Naik, A., Stap, D. and Pathak, E., 2022. Benchmarking generalization via in-context instructions on 1,600+ language tasks. arXiv preprint arXiv:2204.07705.
3. Kosuke Nishida, Kyosuke Nishida, and Sen Yoshida. 2021. Task-adaptive Pre-training of Language Models with Word Embedding Regularization. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 4546–4553, Online. Association for Computational Linguistics.
4. Schick, T., & Schütze, H. (2020). Rare Words: A Major Problem for Contextualized Embeddings and How to Fix it by Attentive Mimicking. Proceedings of the AAAI Conference on Artificial Intelligence, 34(05), 8766-8774. https://doi.org/10.1609/aaai.v34i05.6403
5. Wang H, Liu C, Xi N, Zhao S, Ju M, Zhang S, Zhang Z, Zheng Y, Qin B, Liu T. Prompt Combines Paraphrase: Teaching Pre-trained Models to Understand Rare Biomedical Words. arXiv preprint arXiv:2209.06453. 2022 Sep 14.
6. Grundmann, Paul, Sebastian Arnold, and Alexander Löser. "Self-supervised answer retrieval on clinical notes." arXiv preprint arXiv:2108.00775 (2021).

