from typing import Dict, List
from collections import defaultdict,Counter
import math
from util import get_char_ngrams, argmax
import pdb



class NBLangIDModel:
    def __init__(self, ngram_size: int = 2, extension: bool = False):
        """
        NBLangIDModel constructor

        Args:
            ngram_size (int, optional): size of char n-grams. Defaults to 2.
            extension (bool, optional): set to True to use extension code. Defaults to False.
        """
        self._priors = {}
        self._likelihoods = {}
        self.ngram_size = ngram_size
        self.extension = extension
        self._vocab = set()

    def fit(self, train_sentences: List[str], train_labels: List[str]):
        """
        Train the Naive Bayes model (by setting self._priors and self._likelihoods)

        Args:
            train_sentences (List[str]): sentences from the training data
            train_labels (List[str]): labels from the training data
        """
        self._priors = {}
        for label in train_labels:
            self._priors[label] = self._priors.get(label, 0) + 1
        
        for label in train_labels:
            count = train_labels.count(label)
            self._priors[label] = math.log(count/ len(train_labels))
          
        #Uniform priors
        #for label in train_labels:
            #self._priors[label] = math.log(1/len(train_labels))

        ngram_counts = defaultdict(lambda: Counter({ngram: 0 for ngram in self._vocab}))

        for i in train_sentences:
            self._vocab.update(get_char_ngrams(i, self.ngram_size))
    #Raw Ngram count calculation
        for i, label in zip(train_sentences, train_labels):
            ngrams = get_char_ngrams(i, self.ngram_size)
            for ngram in ngrams:
                #breakpoint()
                # Increment count for ngram in the inner dictionary corresponding to the label
                ngram_counts[label][ngram] += 1 
              
        #Laplace add one smoothing
        for ngram in self._vocab:
            for label in ngram_counts:
                ngram_counts[label][ngram]+=1
          

        #key and values of ngram.items is stored in var key and value.
        for key, values in ngram_counts.items():
            count_ngram = sum(values.values())
            self._likelihoods.setdefault(key, {})
            for value in values:
                self._likelihoods[key][value] = math.log((values[value]) / (count_ngram) )


    def predict(self, test_sentences: List[str]) -> List[str]:
        """
        Predict labels for a list of sentences

        Args:
            test_sentences (List[str]): the sentence to predict the language of

        Returns:
            List[str]: the predicted languages (in the same order)
        """
        predictions = []
        for c in test_sentences:
            log_probs = self.predict_one_log_proba(c)
            predicted = argmax(log_probs)
            predictions.append(predicted)
        return predictions

    def predict_one_log_proba(self, test_sentence: str) -> Dict[str, float]:
        """
        Computes the log probability of a single sentence being associated with each language

        Args:
            test_sentence (str): the sentence to predict the language of

        Returns:
            Dict[str, float]: mapping of language --> probability
        """
        assert not (self._priors is None or self._likelihoods is None), \
            "Cannot predict without a model!"
    
        log_prob = {}
        ngrams = get_char_ngrams(test_sentence, self.ngram_size)
        for lang in self._likelihoods:
            log_prob[lang] = self._priors[lang]  # Initialize log probability for the language
            for key, value in self._likelihoods[lang].items():
                if key in ngrams:
                    num_occurences = test_sentence.count(key)
                    log_prob[lang] += num_occurences * value  # Accumulate log probability for each ngram
        return log_prob