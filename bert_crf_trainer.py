from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import DebertaConfig, DebertaTokenizerFast, BertConfig, AutoTokenizer, BertTokenizerFast, Trainer, TrainingArguments, DataCollatorForTokenClassification
from transformers.trainer_utils import IntervalStrategy
import numpy as np
import json
import matplotlib.pyplot as plt
from models import BertCRF
from deberta_model import DebertaCRF
import pandas as pd
import os

def untokenize_labels_predictions(word_ids, true_labels, predictions):
    untokenized_true_labels = []
    untokenized_predictions = []

    for sublist_word_ids, sublist_true_labels, sublist_predictions in zip(word_ids, true_labels, predictions):
        current_labels = []
        current_predictions = []
        last_word_id = None

        for word_id, label, prediction in zip(sublist_word_ids[1:-1], sublist_true_labels, sublist_predictions):
            # Skip if this word_id is the same as the last one (it's a subword)
            if word_id == last_word_id:
                continue

            current_labels.append(label)
            current_predictions.append(prediction)
            last_word_id = word_id

        untokenized_true_labels.append(current_labels)
        untokenized_predictions.append(current_predictions)

    return untokenized_true_labels, untokenized_predictions

class TrainingMonitor:
    def __init__(self):
        self.best_f1 = 0
        self.best_confusion_matrix = None

    def compute_metrics_factory(self, word_ids, fold_no, eval_dataset_name, model_name):
        # Define the actual compute_metrics function
        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)

            # Remove ignored index (special tokens) and convert to labels
            true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
            true_predictions = [
                [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            untokenized_true_labels, untokenized_predictions = untokenize_labels_predictions(word_ids, true_labels, true_predictions)

            unflat_true = [label for seq in untokenized_true_labels for label in seq]
            unflat_pred = [label for seq in untokenized_predictions for label in seq]
            unreport = classification_report(y_pred=unflat_pred, y_true=unflat_true, output_dict=True)
            unreport['macro_wo_O'] = {'precision': (unreport['I-Background']['precision'] + unreport['I-Other']['precision'] + unreport['I-Problem']['precision'] + unreport['I-Test']['precision'] + unreport['I-Treatment']['precision']) / 5,
            'recall': (unreport['I-Background']['recall'] + unreport['I-Other']['recall'] + unreport['I-Problem']['recall'] + unreport['I-Test']['recall'] + unreport['I-Treatment']['recall']) / 5,
            'f1-score': (unreport['I-Background']['f1-score'] + unreport['I-Other']['f1-score'] + unreport['I-Problem']['f1-score'] + unreport['I-Test']['f1-score'] + unreport['I-Treatment']['f1-score']) / 5,
            'support': (unreport['I-Background']['support'] + unreport['I-Other']['support'] + unreport['I-Problem']['support'] + unreport['I-Test']['support'] + unreport['I-Treatment']['support'])}
            
            un_report_df = pd.DataFrame(unreport).round(3).T

            new_f1_score = unreport['macro_wo_O']['f1-score']
            if self.best_f1 < new_f1_score:
                self.best_f1 = new_f1_score
                cm = confusion_matrix(y_pred=unflat_pred, y_true=unflat_true)
                disp = ConfusionMatrixDisplay(cm, display_labels=np.array(['I-Background','I-Other', 'I-Problem', 'I-Test', 'I-Treatment', 'O']))
                fig, ax = plt.subplots(figsize=(8, 8))
                disp.plot(ax=ax)

              
                binary_predictions = ['0' if label == 'O' else '1' for label in unflat_pred]
                binary_labels = ['0' if label == 'O' else '1' for label in unflat_true]

                # Generate a classification report
                binary_classification_report = classification_report(y_true=binary_labels, y_pred=binary_predictions, target_names=['O', 'I'], digits=3, output_dict=True)
                
                # # Save the figure to an image file
                # plt.savefig(f'analysis/{model_name}/{dataset_name}/graphs/fold{fold_no}/confusion_matrix.png')
                # with open(f"analysis/{model_name}/{dataset_name}/reports/fold{fold_no}/multiclass_classification_report.json", "w") as f:
                #     json.dump(un_report_df.to_dict(), f, indent=4)
                # with open(f"analysis/{model_name}/{dataset_name}/reports/fold{fold_no}/binary_classification_report.json", "w") as f:
                #     json.dump(binary_classification_report, f, indent=4)

                # Define the directory paths
                graph_dir = f'analysis/{model_name}/{eval_dataset_name}/graphs/fold{fold_no}'
                report_dir = f'analysis/{model_name}/{eval_dataset_name}/reports/fold{fold_no}'

                # Create directories if they do not exist
                os.makedirs(graph_dir, exist_ok=True)
                os.makedirs(report_dir, exist_ok=True)

                # Save the confusion matrix as a PNG file
                plt.savefig(f'{graph_dir}/confusion_matrix.png')
                plt.close()

                # Save the multiclass classification report as a JSON file
                with open(f"{report_dir}/multiclass_classification_report.json", "w") as f:
                    json.dump(un_report_df.to_dict(), f, indent=4)

                # Save the binary classification report as a JSON file
                with open(f"{report_dir}/binary_classification_report.json", "w") as f:
                    json.dump(binary_classification_report, f, indent=4)
                

            return {
                "precision": unreport['macro_wo_O']['precision'],
                "recall": unreport['macro_wo_O']['recall'],
                "f1": unreport['macro_wo_O']['f1-score'],
                "accuracy": unreport['accuracy'],
            }
        return compute_metrics

def tokenize(batch):
  result = {
      'label_ids': [],
      'input_ids': [],
      'token_type_ids': [],
  }
  max_length = tokenizer.model_max_length

  for tokens, label in zip(batch['tokens'], batch['label_ids']):
      tokenids = tokenizer(tokens, add_special_tokens=False)

      token_ids = []
      label_ids = []
      for ids, lab in zip(tokenids['input_ids'], label):
          # print(ids)
          # print(lab)
        # Extend token_ids with the current token or subword tokens
          token_ids.extend(ids)
          # print(token_ids)
          # If the token is split into subwords, copy the label for each subword token
          chunk = [lab] * len(ids)
          
          # Extend label_ids with the labels for each token or subword token
          label_ids.extend(chunk)

      token_type_ids = tokenizer.create_token_type_ids_from_sequences(token_ids)
      token_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
      label_ids.insert(0, 0)
      label_ids.append(0)
      result['input_ids'].append(token_ids)
      result['label_ids'].append(label_ids)
      result['token_type_ids'].append(token_type_ids)

  result = tokenizer.pad(result, padding='longest', max_length=max_length, return_attention_mask=True)
  for i in range(len(result['input_ids'])):
      diff = len(result['input_ids'][i]) - len(result['label_ids'][i])
      result['label_ids'][i] += [0] * diff
  return result

# Split the dataset manually
# train_test_split = full_dataset['train'].select(range(20)).train_test_split(test_size=0.2)
# train_dataset = train_test_split['train']
# test_dataset = train_test_split['test']

# train_dataset, test_dataset = load_dataset('conll2003', split=['train', 'test'])

model_names = ['bluebert-ft'] #['scibert-ft', 'bluebert-ft', 'bertclinical-ft', 'bioclinicalbert-ft', 'deberta-ft']
model_checkpoints = ['bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16'] #['allenai/scibert_scivocab_uncased', 'bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16', 'samrawal/bert-base-uncased_clinical-ner' ,'emilyalsentzer/Bio_ClinicalBERT', 'microsoft/deberta-base']
dataset_name = 'phee'
eval_dataset_name = 'phee'
for model_name, model_checkpoint in zip(model_names, model_checkpoints):
  if model_checkpoint == 'microsoft/deberta-base':
    tokenizer = DebertaTokenizerFast.from_pretrained(model_checkpoint, add_prefix_space=True)
  else:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
  # data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

  id2label = {0: 'O', 1: 'I-Treatment', 2: 'I-Test', 3: 'I-Problem', 4: 'I-Background', 5: 'I-Other'}


  for i in range(1):

    dataset = load_dataset('json', field='data', data_files={'train': f'data/processed/{dataset_name}/fold{i}/train.json', 'test': f'data/processed/{eval_dataset_name}/fold{i}/test.json'})
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    test_sentences = test_dataset['tokens']
    test_word_ids = []
    for sentence in test_sentences:
        test_word_ids.append(tokenizer(sentence, truncation=True, is_split_into_words=True).word_ids())

    train_dataset = train_dataset.rename_column('ner_tags', 'label_ids')
    test_dataset = test_dataset.rename_column('ner_tags', 'label_ids')

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))
    
    train_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label_ids'])
    test_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label_ids'])

    num_labels=6
    # model = BertCRF.from_pretrained(model_checkpoint, num_labels=6, ignore_mismatched_sizes=True)
    if model_checkpoint == 'microsoft/deberta-base':
      config = DebertaConfig.from_pretrained(model_checkpoint)
      config.num_labels = num_labels
      model = DebertaCRF.from_pretrained(model_checkpoint, config=config, ignore_mismatched_sizes=True)
    else:
      config = BertConfig.from_pretrained(model_checkpoint)
      config.num_labels = num_labels
      model = BertCRF.from_pretrained(model_checkpoint, config=config, ignore_mismatched_sizes=True)

    monitor = TrainingMonitor()
    training_args = TrainingArguments(
        output_dir=f"models/{model_name}/fold{i}",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        metric_for_best_model="f1",
        save_strategy="epoch",
        learning_rate=1e-5,
        warmup_steps=200,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=monitor.compute_metrics_factory(test_word_ids, fold_no=i, eval_dataset_name=eval_dataset_name, model_name = model_name),
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

    trainer.evaluate()
