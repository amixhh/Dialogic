import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
from tqdm import tqdm
import os

class ClassificationEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        
    def calculate_bleu(self, reference, candidate):
        """Calculate BLEU score between reference and candidate text"""
        smoothie = SmoothingFunction().method1
        return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)
    
    def calculate_rouge(self, reference, candidate):
        """Calculate ROUGE scores between reference and candidate text"""
        scores = self.scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def calculate_bertscore(self, reference, candidate):
        """Calculate BERTScore between reference and candidate text"""
        P, R, F1 = self.bert_scorer.score([candidate], [reference])
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
    
    def calculate_classification_metrics(self, true_topic, pred_topic, true_subtopic, pred_subtopic):
        """Calculate classification metrics (accuracy only per-sample)"""
        return {
            'topic_accuracy': 1.0 if true_topic == pred_topic else 0.0,
            'true_topic': true_topic,
            'pred_topic': pred_topic,
            'true_subtopic': true_subtopic,
            'pred_subtopic': pred_subtopic
        }
    
    def evaluate_on_dataset(self, dataset_path, classify_function):
        """Evaluate classification performance on a test dataset"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]
        
        results = {
            'text_similarity': [],
            'classification': []
        }
        
        for item in tqdm(test_data, desc="Evaluating"):
            true_topic = item['topic']
            true_subtopic = item.get('subtopic')
            true_text = item['text']

            pred_result = classify_function(true_text)
            pred_topic = pred_result['topic']
            pred_subtopic = pred_result.get('subtopic')

            text_metrics = {
                'bleu': self.calculate_bleu(true_text, true_text),
                'rouge': self.calculate_rouge(true_text, true_text),
                'bertscore': self.calculate_bertscore(true_text, true_text)
            }
            results['text_similarity'].append(text_metrics)

            classification_metrics = self.calculate_classification_metrics(
                true_topic, pred_topic, true_subtopic, pred_subtopic
            )
            results['classification'].append(classification_metrics)

        true_topics = [r['true_topic'] for r in results['classification']]
        pred_topics = [r['pred_topic'] for r in results['classification']]

        topic_precision, topic_recall, topic_f1, _ = precision_recall_fscore_support(
            true_topics, pred_topics, average='weighted', zero_division=0
        )

        aggregated_results = {
            'text_similarity': {
                'bleu': np.mean([r['bleu'] for r in results['text_similarity']]),
                'rouge1': np.mean([r['rouge']['rouge1'] for r in results['text_similarity']]),
                'rouge2': np.mean([r['rouge']['rouge2'] for r in results['text_similarity']]),
                'rougeL': np.mean([r['rouge']['rougeL'] for r in results['text_similarity']]),
                'bertscore_precision': np.mean([r['bertscore']['precision'] for r in results['text_similarity']]),
                'bertscore_recall': np.mean([r['bertscore']['recall'] for r in results['text_similarity']]),
                'bertscore_f1': np.mean([r['bertscore']['f1'] for r in results['text_similarity']])
            },
            'classification': {
                'topic_accuracy': np.mean([r['topic_accuracy'] for r in results['classification']]),
                'topic_precision': topic_precision,
                'topic_recall': topic_recall,
                'topic_f1': topic_f1,
            }
        }
        
        return aggregated_results

def main():
    from classify_query import classify_query
    
    evaluator = ClassificationEvaluator()
    dataset_path = "data/test_dataset.jsonl"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Test dataset not found at {dataset_path}")
        return
    
    results = evaluator.evaluate_on_dataset(dataset_path, classify_query)
    
    print("\nEvaluation Results:")
    print("\nText Similarity Metrics:")
    for metric, value in results['text_similarity'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nClassification Metrics:")
    for metric, value in results['classification'].items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()