
# Sentiment Analysis 
![NLP](https://img.shields.io/badge/NLP-1F425F?style=for-the-badge)
![BERT](https://img.shields.io/badge/BERT-3C9C8A?style=for-the-badge)


# Sentiment Analysis with BERT

This project harnesses the power of sentiment analysis to evaluate user reviews, predicting sentiments as either positive or negative using the advanced BERT (Bidirectional Encoder Representations from Transformers) model.

## 🚀 What is Sentiment Analysis?

Sentiment analysis involves deciphering the sentiment or emotion conveyed in a piece of text, be it positive, negative, or neutral. In this venture, sentiment analysis is applied to classify user reviews into positive or negative categories.

## 🤖 What is BERT?

BERT, short for Bidirectional Encoder Representations from Transformers, stands as a transformer-based machine-learning model crafted by Google. Its prowess lies in capturing bi-directional context within a given text, making it exceptionally adept at tasks like sentiment analysis.

The model undergoes pre-training on an extensive corpus of text data, allowing it to grasp the context and nuances of words across various scenarios. This pre-training empowers BERT to subsequently undergo fine-tuning for specialized tasks like text classification, question-answering, and more.

In simpler terms, BERT operates as a super-smart language understanding tool, reading and comprehending text akin to human understanding. This versatility renders it invaluable for diverse natural language processing tasks, encompassing sentiment analysis, language translation, and information retrieval.

## 💡 Why BERT?

BERT is the chosen sentinel for sentiment analysis due to its unparalleled ability to fathom the context and essence of words within a sentence. This surpasses the capabilities of traditional machine learning models, making BERT exceptionally fitting for sentiment analysis tasks where contextual comprehension is paramount for precise classification.






### 📚 Libraries Used
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)

![Seaborn](https://img.shields.io/badge/Seaborn-3776ab?style=for-the-badge&logo=seaborn&logoColor=white)

![Transformers](https://img.shields.io/badge/Transformers-FFAC45?style=for-the-badge&logo=transformers&logoColor=white)

![Torch](https://img.shields.io/badge/Torch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

![Requests](https://img.shields.io/badge/Requests-2CA5E0?style=for-the-badge&logo=requests&logoColor=white)

<p>⚙️ Deployment</p>

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

# 🔄 Sentiment Analysis Workflow

* Data Preprocessing: Raw text data is preprocessed and tokenized using the BERT tokenizer.
* Model Training: The BERT model is utilized for training on labeled data to learn the sentiment patterns.
* Prediction: The trained model is used to predict the sentiment (positive or negative or neutral) of new input text based on learned patterns.




1. **Input:**
   - The input is the text data you want to perform sentiment analysis on. This text data is what you want BERT to analyze and make predictions about.

2. **Token:**
   - In NLP, a token is a unit of text, typically a word or a subword. BERT tokenizes your input text into these smaller units, which helps the model understand the text better.

3. **BERT Cased Tokenization:**
   - BERT uses a specific tokenization method, often called "WordPiece" tokenization. It tokenizes text into subword pieces, including both lowercase and uppercase versions. This helps capture case-related information.

4. **Model Training:**
   - This refers to the process where the BERT model is trained on a large corpus of text data. During this phase, BERT learns to predict missing words (Masked Language Model, MLM) and understand sentence relationships (Next Sentence Prediction, NSP).

5. **Pre-training (MLM, NSP):**
   - BERT is pre-trained on a vast amount of text data using two tasks - MLM, where it predicts masked words in a sentence, and NSP, where it predicts whether two sentences are contiguous in the original text. This pre-training helps BERT capture general language understanding.

6. **Fine-tuning:**
   - This is where you adapt the pre-trained BERT model to your specific task. You fine-tune the model on a labeled dataset for sentiment analysis.

7. **Classifier Layer:**
   - During fine-tuning, you typically add a classifier layer on top of the BERT model. This layer maps the BERT model's output to sentiment labels (positive, negative, neutral) in your case.

8. **CLS, SEP:**
   - In BERT, [CLS] and [SEP] tokens are special tokens used to denote the beginning and separation of sentences. The [CLS] token is particularly important, as it is used for classification tasks. It encapsulates information about the entire input sequence.

9. **Output: Positive, Negative, Neutral:**
   - The output refers to the predictions made by the BERT model. It will assign sentiment labels to the input text, such as "positive," "negative," or "neutral," based on the training it received.







## ✨ Features

1. Contextual Understanding
2. Nuanced Sentiment Analysis
3. Ambiguity Resolution
4. Domain-Specific Fine-Tuning
5. Enhanced Accuracy  

## ▶️ How to Run the Project

To run this project, follow these steps:

1. **Ensure Both CSV and Notebook are Present:**
   - Make sure that both the CSV file containing your data and the Jupyter Notebook (or any other notebook file) are located in the same directory.

2. **Open the Notebook with Any Editor:**
   - Open the Jupyter Notebook using any code editor or integrated development environment (IDE) of your choice. You can use editors like Jupyter, VSCode, or any other notebook-compatible environment.

3. **Run the Cells:**
   - Navigate through the notebook and run each cell sequentially. This can typically be done by selecting a cell and either clicking the "Run" button or using the keyboard shortcut (usually Shift + Enter).

4. **Review Output:**
   - After running all the cells, review the output and any generated visualizations or results within the notebook.


### 📝 To check the sentiment within the text, pass the text as a parameter to this function.

```
prediction_on_raw_data(raw_text)

```
| Label    |
|----------|
| Positive Text |
| Negative Text |
| Neutral Text  |


## 📑 License

[![MIT License](https://img.shields.io/badge/License-MIT-0178B5?style=for-the-badge)](https://github.com/Kammarianand/Sentiment-Analysis-BERT/blob/main/LICENSE)
