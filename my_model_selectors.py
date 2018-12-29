import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        bic_min, model_best = None, None
        for x in range(self.min_n_components, self.max_n_components + 1):
            model_fit = self.base_model(x)
            try:
                log_like = model_fit.score(self.X, self.lengths)
                n_features = len(self.X[0])
                # Calculate p in the formula above.
                parameters = x**2 + 2 * x * n_features - 1
                # Calculate logN in the formula above
                log_points = math.log(len(self.X))
                bic = -2 * log_like + parameters * log_points
                if bic_min is None or bic < bic_min:
                    bic_min = bic
                    model_best = model_fit
            except:
                pass
        return model_best


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        dic_min, model_best = None, None
        for x in range(self.min_n_components, self.max_n_components + 1):
            model_fit = self.base_model(x)
            try:
                # calculate the score for the data
                log_like = model_fit.score(self.X, self.lengths)
                # Need to calculate the anti-likelihood
                log_like_else = 0
                for word in self.words.keys():
                    if word != self.this_word:
                        other_X, other_len = self.hwords[word]
                        # Get the score for the competing category
                        score = model_fit.score(other_X, other_len)
                        # add the score to the anti-likelihood
                        log_like_else += score

                n_words = len(self.words.keys())
                # formula (18)
                dic = log_like - (1/(n_words - 1)*log_like_else)
                if dic_min is None or dic > dic_min:
                    dic_min = dic
                    model_best = model_fit
            except:
                pass
        return model_best

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        split_method = KFold()
        n_components = len(self.sequences)
        ll_max, model_best = None, None
        # use cv when possible
        if n_components > 2:
            for x in range(self.min_n_components, self.max_n_components + 1):
                avg_log_like = 0
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    train_X, train_len = combine_sequences(cv_train_idx,
                                                           self.sequences)
                    test_X, test_len = combine_sequences(cv_test_idx,
                                                         self.sequences)
                    self.X = train_X
                    self.lengths = train_len
                    model_fit = self.base_model(x)
                    try:
                        log_like = model_fit.score(test_X, test_len)
                        avg_log_like += log_like
                    except:
                        pass
                # average the likelihood scores over the cv splits
                avg_log_like /= 3
                # Update if theirs a new max
                if ll_max is None or ll_max < avg_log_like:
                    ll_max = avg_log_like
                    model_best = model_fit
            return model_best
        # otherwise
        for x in range(self.min_n_components, self.max_n_components + 1):
            model_fit = self.base_model(x)
            try:
                log_like = model_fit.score(self.X, self.lengths)
                if ll_max is None or ll_max < log_like:
                    ll_max = log_like
                    model_best = model_fit
            except:
                pass
        return model_best
