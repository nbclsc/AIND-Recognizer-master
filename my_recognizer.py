import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    probabilities = list()
    guesses = list()
    stuff = test_set.get_all_Xlengths()
    # Loop through new testing data.
    for index_key, test_word in stuff.items():
        # For each testing word, fit for each model saving log-likelihood value.
        test_scores = dict()
        # Loop through the models.
        for word, model in models.items():
            try:
                test_X = test_word[0]
                test_len = test_word[1]
                # Get that juicy score and add to a dictionary for probs.
                test_log_like = model.score(test_X, test_len)
                test_scores[word] = test_log_like
            except:
                pass
        # Find the best guess word.
        best_guess = max(test_scores, key=lambda key: test_scores[key])
        guesses.append(best_guess)
        probabilities.append(test_scores)
    return probabilities, guesses
