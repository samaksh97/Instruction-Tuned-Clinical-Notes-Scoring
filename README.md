# Instruction-Tuned-Clinical-Notes-Scoring
Instruction Tuned Scoring using Clinical Notes 


Introduction:

Our criteria while searching problem statements included 
a. A positive impact in the healthcare system 
b. Problems where NLP was a potential solution.
This was primarily driven by the advances in NLP that have created significant impact in healthcare for both patients and physicians.[1]. 
With these guidelines we narrowed down to a problem statement related to reduce the burden of .
Recently introduced instruction-paradigm empowers non-expert users to leverage NLP resources by defining a new task in natural language. Instruction-tuned models have the potential to significantly outperform task-specific language models (without instruction), but their effect has not yet been systematically studied in the healthcare domain. Also, Large Language models have been trained on a general English corpus and may not perform well on domain-specific domains which have a high proportion of rare words. 

The database presents a corpus of 43,985 clinical patient notes (PNs) written by 35,156 examinees during the high-stakes USMLE® Step 2 Clinical Skills examination. 
In medical education, students are often assessed through encounters with standardized patients - people trained to portray simulated scenarios called clinical cases. For each such encounter, the student is expected to perform a history and physical examination, determine differential diagnoses, and then document their findings in a PN. 

++ Dataset
++features



Problem Definition:
USMLE® Step 2 Clinical Skills examination is one of the most prestigious and rigorous medical licensure exams for medical professionals. The exam required test-takers to interact with Standardized Patients (people trained to portray specific clinical cases) and write a patient note. Trained physician raters later scored patient notes with rubrics that outlined each case’s important concepts (referred to as features). 

The goal of our project is to develop an automated way of identifying the relevant features within each patient note,with a special focus on the patient history portions of the notes where the information from the interview with the standardized patient is documented. 
Methods:
When working on a specific domain, it is ideal to pre-train the LLM on rare words first and then perform the domain-specific task. Here, we plan to first pre-train the instruction-tuned T5 LLM so that it learns the rare biomedical terms. We will then fine-tune it using Instructional prompts composed of a definition and a positive example. 

Potential Results and Discussions:
As our aim is to identify semantically similar phrases from the patient notes that match the medically termed features, we plan to use micro-averaged F1 score to evaluate the overlap of the predicted phrase spans with the span of the labels.

At least three references:
Mihir Parmar, Swaroop Mishra, Mirali Purohit, Man Luo, Murad Mohammad, and Chitta Baral. 2022. In-BoXBART: Get Instructions into Biomedical Multi-Task Learning. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 112–128, Seattle, United States. Association for Computational Linguistics.
Wang, Y., Mishra, S., Alipoormolabashi, P., Kordi, Y., Mirzaei, A., Arunkumar, A., Ashok, A., Dhanasekaran, A.S., Naik, A., Stap, D. and Pathak, E., 2022. Benchmarking generalization via in-context instructions on 1,600+ language tasks. arXiv preprint arXiv:2204.07705.
Puri, R.S., Mishra, S., Parmar, M. and Baral, C., 2022. How Many Data Samples is an Additional Instruction Worth?. arXiv preprint arXiv:2203.09161.


