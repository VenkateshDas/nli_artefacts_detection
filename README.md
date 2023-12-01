# SNLI Artefacts detection

---

This notebook presents an implementation of an analysis method for detecting and understanding artefacts in NLP datasets, specifically the Stanford Natural Language Inference (SNLI) dataset. This analysis is based on the research paper titled **"Competency Problems: On Finding and Removing Artifacts in Language Data"**.

In NLP tasks, datasets often contain subtle biases or artefacts that models may exploit to make predictions. These biases aren't always evident upon cursory inspection and may lead to overestimated performance metrics as models rely on them rather than learning the intended linguistic phenomena.

The research paper this notebook refers to proposes a method for detecting these dataset artefacts using statistical techniques. **The approach is grounded on the calculation of Z-statistics for each token in the dataset, considering each label. A token with a Z-statistic that significantly deviates from what we would expect under a null hypothesis is considered a potential artefact.**

***This notebook covers the process of loading and preprocessing the dataset, calculating token statistics, and using these statistics to identify potential artefacts.*** It also provides a visualization of these artefacts, offering insights into how they distribute across different labels in the dataset.


---

---
Artefacts Graph from the Paper</br>
![Artefacts Graph from the Paper](https://drive.google.com/uc?export=view&id=19MS7GKzXYBkjIQDlyDQKWIET0XFjSGmX)

Reproduced Artefacts Graph from the paper
![Reproduced artefacts from the paper](https://drive.google.com/uc?export=view&id=12gg3hJVXboRYRLhjDGwvgmT2UcTU4stU)

---



---

Note : The artefacts detection method was implemented based on the information provided on the paper.
But the number of tokens in the vocabulary is not similar to the one mentioned in the paper. This might be due to the different preprocessing applied. I tried to remove the punctuations and split the sentence on whitespaces.

---

Please refer the notebook for further calculations.

---

### References

Relevant sections from the paper : "Competency Problems: On Finding and Removing Artifacts in Language Data" 

1.   Section 2 - Competency Problems
2.   Section 3.2 Hypothesis Test
3.   Section 4.1 Data Analysis
4.   Section 4.2 Model Analysis
