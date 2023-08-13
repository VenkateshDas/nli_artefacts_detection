from transformers import Trainer, TrainingArguments, pipeline, AutoModelForSequenceClassification, AutoTokenizer, BertTokenizerFast
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from datasets import Dataset
from transformers import DataCollatorWithPadding
import torch
import pandas as pd
import numpy as np
import evaluate
import os
from collections import Counter



class Baseline:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self, y_pred):
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

class MajorityBaseline(Baseline):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.X_train = pd.DataFrame({'dummy': [0]*len(y_train)})
        self.X_test = pd.DataFrame({'dummy': [0]*len(y_test)})


    def run_baseline(self):
        # Create a dummy classifier that will always predict the majority class
        majority_class = Counter(list(self.y_train)).most_common(1)[0][0]
        y_pred = [majority_class] * len(self.y_test)
        # clf = DummyClassifier(strategy='most_frequent')
        # clf.fit(self.X_train, self.y_train)
        # y_pred = clf.predict(self.X_test)
        accuracy = evaluate.load("accuracy").compute(predictions=y_pred, references=self.y_test)
        # accuracy = self.evaluate(y_pred)
        return accuracy

class HypothesisOnlyBaseline(Baseline):
    def __init__(self, X_train, y_train, X_test, y_test, model_name, epochs, device, label_dict, id2label):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, id2label=id2label, label2id=label_dict).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.device = device
        self.epochs = epochs
        
    
    def compute_metrics(self, eval_pred):
        accuracy = evaluate.load("accuracy")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    def train_model(self):
        training_args = TrainingArguments(
            output_dir='./results',
            learning_rate=2e-5,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            use_mps_device=False,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.X_train,
            eval_dataset=self.X_test,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()

    def infer_model(self, model_path, sentences):
        # model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        # infer_pipeline = pipeline('text-classification', model=model, tokenizer=self.tokenizer, device=self.device)
        # predictions = infer_pipeline(sentences)
        # y_pred = [pred['label'] for pred in predictions]
         # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        # tok_sentences = self.tokenizer(sentences, truncation=True)
        # self.trainer.evaluate(self.X_test)
        label_dict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        id2label = {0:'entailment', 1:'neutral', 2:'contradiction'}
        infer_pipeline = pipeline('text-classification', model=model, tokenizer=self.tokenizer, device=self.device)
        predictions = infer_pipeline(sentences)
        preds = [label_dict[pred['label']] for pred in predictions]

        # # Predict the labels
        # predictions = model(tok_sentences)

        # # Convert the predictions to a NumPy array
        # predictions = predictions.cpu().numpy()
        # # return y_pred
        return preds


    def run_baseline(self, model_path, sentences):
        # Train the model on the training data
        print("Training")
        self.train_model()

        # Get the predictions of the model on the test data
        print("Testing with only Hypothesis")
        y_pred = self.infer_model(model_path, sentences)

        # # Calculate the accuracy of the predictions
        # accuracy = self.evaluate(y_pred)
        accuracy = evaluate.load("accuracy")
        acc = accuracy.compute(predictions=y_pred, references=self.y_test)

       

        return acc

def load_and_preprocess_data(source, model_name, device='cpu'):
    """
    Load and preprocess the NLI data from either a local CSV file or a HuggingFace dataset.
    The CSV file or HuggingFace dataset is expected to have the following columns: "premise", "hypothesis", "label".
    """
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if source.startswith('csv:'):
        # Load the data from a CSV file
        filepath = source[4:]
        df = pd.read_csv(filepath)
        label_dict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        df['label'] = df['label'].map(label_dict)
        X_train, X_test, y_train, y_test = train_test_split(df[['premise', 'hypothesis']], df['label'], test_size=0.2, random_state=42)

        # Tokenize the data
        X_train = tokenizer(list(X_train['premise']), list(X_train['hypothesis']), truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
        X_test = tokenizer(list(X_test['premise']), list(X_test['hypothesis']), truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
    else:
        # Load the data from a HuggingFace dataset
        dataset = load_dataset(source)

        # Define a helper function for tokenization
        def tokenize_train(batch):
            return tokenizer(batch['premise'], batch['hypothesis'], truncation=True)

        # Define a helper function for tokenization
        def tokenize_test(batch):
            return tokenizer(batch['hypothesis'], padding='max_length', truncation=True)

        # Process the dataset
        train_dataset = dataset['train'][:]
        test_dataset = dataset['test'][:] 
        train_df = pd.DataFrame(train_dataset)
        eval_df = pd.DataFrame(dataset['validation'])[:]
        train_df = train_df[train_df['label'] != -1]
        eval_df = eval_df[eval_df['label'] != -1]
        test_df = pd.DataFrame(test_dataset)
        test_df = test_df[test_df['label'] != -1]
        hyp_only_test_df = test_df.drop(columns=['premise'])
        hypothesis_sentences = hyp_only_test_df['hypothesis']
        train_dataset = Dataset.from_pandas(train_df)
        hyp_only_test_dataset = Dataset.from_pandas(hyp_only_test_df)
        eval_dataset = Dataset.from_pandas(eval_df)

        # Tokenize the data
        train_dataset = train_dataset.map(tokenize_train, batched=True)
        eval_dataset = eval_dataset.map(tokenize_train, batched=True)
        hyp_only_test_dataset = hyp_only_test_dataset.map(tokenize_test, batched=True)

        # Set the format
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'], device=device)
        hyp_only_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'], device=device)
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'], device=device)

        # Get the train and test splits
        X_train, X_test, X_eval = train_dataset, hyp_only_test_dataset, eval_dataset
        y_train = train_dataset['label']
        y_test = hyp_only_test_dataset['label']
        y_eval = eval_dataset['label']

    return X_train, y_train, X_test, y_test, X_eval, y_eval, hypothesis_sentences


def main():
    accuracy = evaluate.load("accuracy")
    model_name = 'bert-base-uncased'
    # source = 'csv:/path/to/your/file.csv'  # for own dataset
    source = 'snli'  # for HuggingFace dataset
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda:0')

    label_dict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    id2label = {0:'entailment', 1:'neutral', 2:'contradiction'}

    print("Loading dataset and preprocessing them...")
    X_train, y_train, X_test, y_test, _, _, hyp_sentences = load_and_preprocess_data(source, model_name, device)
    # X_test = preprocess_hypothesis(X_test)

    # Majority Baseline
    majority_baseline = MajorityBaseline(X_train, y_train, X_test, y_test)
    majority_accuracy = majority_baseline.run_baseline()
    print("Majority Baseline Accuracy:", majority_accuracy)

    # Hypothesis Only Baseline
    epochs = 1
    hypothesis_baseline = HypothesisOnlyBaseline(X_train, y_train, X_test, y_test, model_name, epochs, device, label_dict, id2label)
    accuracy = hypothesis_baseline.run_baseline(model_path="/content/results/checkpoint-7", sentences=list(hyp_sentences))
    print(f'Hypothesis-only baseline accuracy: {accuracy}')

if __name__ == '__main__':
    main()