import xml.etree.ElementTree as ET
import nltk.data
import nltk.tag
import math
from nltk.corpus import stopwords


class NaiveBayesModel:
    """
    Naive Bayes Model for word sense disambiguation

    Authors: Philip Su ps845, Kelvin Jin kkj9
    """
    COLLOCATION = 0
    COOCCURRENCE = 1
    N_CONTEXT = 3
    SMOOTHING_CONSTANT = 1

    def __init__(self):
        """
        Initializes data structures for model. Training data is a list of all training feature
        vectors, categorized by feature type. Sense count map is the count of each sense.
        Total count is the total count of all senses that appear. word_feature_set and
        pos_feature_set are sets of features learned during training.
        """
        self.training_data = {}
        self.sense_count_map = {}
        self.total_count = 0
        self.word_feature_set = set()
        self.pos_feature_set = set()

    def __create_feature_vector(self, context_body, word, feature_type):
        """
        Creates a feature vector of type feature_type given the context and the word.

        :param context_body: text surrounding the word
        :param word: the word the be disambiguated
        :param feature_type: a collocational or coocurence feature
        :return: FeatureVector object
        """
        if feature_type == self.COOCCURRENCE:
            return CooccurrenceFeatureVector(context_body, word)
        elif feature_type == self.COLLOCATION:
            return CollocationFeatureVector(context_body, word)
        else:
            return None

    def add_word_sense(self, senseid):
        """
        Adds a word sense entry to LexeltSenseMap

        :param lexelt: the lexelt object
        :param attrib: the XML attribute of <answer> tag
        """
        # Increment count for a word sense
        if senseid in self.sense_count_map:
            self.sense_count_map[senseid] += 1
        else:
            self.sense_count_map[senseid] = 1
        self.total_count += 1

    def __get_prior(self, senseid):
        """
        Calculate the prior probability of a sense: counts of sense / counts of all senses

        :param senseid: a sense id
        :return: prior probability of a word sense
        """
        if senseid in self.sense_count_map:
            sense_count = self.sense_count_map[senseid]
            return float(sense_count) / self.total_count
        else:
            return 0

    def train(self, context_body, senses, feature_type, word):
        """
        Creates a feature vector from the context of a word for a list of sense that have the
        same context. Adds the feature vector to corresponding feature dictionary.

        :param context_body: text of context around a word
        :param senses: a list of senses associated with this instance
        :param feature_type: what type of feature to train with
        :param word: word to be trained on
        """
        feature_vector = self.__create_feature_vector(context_body, word, feature_type)
        if feature_type not in self.training_data:
            self.training_data[feature_type] = []
        for sense in senses:
            self.training_data[feature_type].append((feature_vector, sense))
        if feature_type == NaiveBayesModel.COLLOCATION:
            for pos_list in feature_vector.get_values():
                for pos in pos_list:
                    self.pos_feature_set.add(pos)
        else:
            for word in feature_vector.get_values():
                self.word_feature_set.add(word)

    def test(self, context_body, word, feature_type):
        """
        Given a context body and a word, calculates the conditional probability of a feature
        given a sense for all senses.

        :param context_body: text of context around a word
        :param word: word to be tested on
        :param feature_type: what type of feature to test with
        :return A dictionary of conditional probabilities and the prior probabilities
        """
        feature_vector = self.__create_feature_vector(context_body, word, feature_type)
        result = {}
        prior = {}
        if feature_type == NaiveBayesModel.COLLOCATION:
            category_dim = len(self.pos_feature_set)
        else:
            category_dim = len(self.word_feature_set)
        for sense in self.sense_count_map:
            prob_f_given_s = feature_vector.get_feature_probability_given_sense(
                self.training_data[feature_type], sense, category_dim)
            result[sense] = prob_f_given_s
            prior[sense] = math.log(self.__get_prior(sense))
        return result, prior


class NaiveBayesModelCatalog:
    """
    A catalog of words to its corresponding Naive Bayes model.

    Authors: Philip Su ps845, Kelvin Jin kkj9
    """
    # A mapping of (word,pos) item to Naive Bayes model
    naive_bayes_models = {}
    feature_types = [NaiveBayesModel.COOCCURRENCE, NaiveBayesModel.COLLOCATION]
    # NLTK's part of speech tagger
    tagger = nltk.data.load(nltk.tag._POS_TAGGER)
    # NLTK's list of stop words
    stop_words = set(stopwords.words('english'))

    # Prevent overfitting by generalizing parts of speeches into Adjectives, Adverbs, Nouns and Verbs
    ADJECTIVE = "ADJ."
    ADVERB = "ADV."
    NOUN = "N."
    VERB = "V."
    pos_map = {
        "JJ": ADJECTIVE,
        "JJR": ADJECTIVE,
        "JJS": ADJECTIVE,
        "NN": NOUN,
        "NNP": NOUN,
        "NNPS": NOUN,
        "NNS": NOUN,
        "PRP": NOUN,
        "PRP$": NOUN,
        "RB": ADVERB,
        "RBR": ADVERB,
        "RBS": ADVERB,
        "VB": VERB,
        "VBD": VERB,
        "VBG": VERB,
        "VBN": VERB,
        "VBP": VERB,
        "VBZ": VERB,
        "WP": NOUN,
        "WP$": NOUN,
        "WRB": ADVERB
    }

    def __init__(self):
        pass

    def train_data_set(self, path_to_train="training-data.data"):
        """
        Initialize data structures with values for Naive Bayes model. Streams the XML file and
        creates a NaiveBayesModel for each word, and adds word senses to NaiveBayesModels as
        well as creates feature vectors.

        :param path_to_train: the path to training data
        """
        # Stream the file in case it won't fit in memory, python's ElementTree
        context = ET.iterparse(path_to_train, events=("start", "end"))
        context = iter(context)
        nb_model = None
        word_form = None
        senses = []
        for event, elem in context:
            if event == "start":
                if elem.tag == "lexelt":
                    word, pos = elem.attrib["item"].split(".")
                    word_pos_pair = (word, pos)
                    if word_pos_pair not in self.naive_bayes_models:
                        # Create Naive Bayes model for word
                        nb_model = NaiveBayesModel()
                        self.naive_bayes_models[word_pos_pair] = nb_model
                    else:
                        # Use Naive Bayes model for word
                        nb_model = self.naive_bayes_models[word_pos_pair]
                elif elem.tag == "answer":
                    # Add list of correct senses to list
                    senses.append(elem.attrib["senseid"])
            elif event == "end":
                if elem.tag == "head":
                    word_form = elem.text
                elif elem.tag == "context":
                    context_body = elem.text
                    # Construct the context_body in case of multiple instances of the word
                    for child in elem:
                        context_body += " " + child.text + " " + child.tail
                    # Train on both features
                    for feature_type in self.feature_types:
                        nb_model.train(context_body, senses, feature_type, word_form)
                    for sense in senses:
                        nb_model.add_word_sense(sense)
                    senses = []

    def validate_data_set(self, path_to_train="training-data-small.data"):
        """
        Initialize data structures with values for Naive Bayes model. Streams the XML file and
        creates a NaiveBayesModel for each word, and adds word senses to NaiveBayesModels as
        well as creates feature vectors. For every 5 instances seen, 4 instances are used to train
        and the last instance is used to predict.

        :param path_to_train: the path to training data

        """
        # Stream the file in case it won't fit in memory
        context = ET.iterparse(path_to_train, events=("start", "end"))
        context = iter(context)
        nb_model = None
        word = None
        word_form = None
        senses = []
        count = 0
        _id = None
        validation_answers = {}
        for event, elem in context:
            if event == "start":
                if elem.tag == "lexelt":
                    word, pos = elem.attrib["item"].split(".")
                    word_pos_pair = (word, pos)
                    if word_pos_pair not in self.naive_bayes_models:
                        nb_model = NaiveBayesModel()
                        self.naive_bayes_models[word_pos_pair] = nb_model
                    else:
                        nb_model = self.naive_bayes_models[word_pos_pair]
                elif elem.tag == "instance":
                    _id = elem.attrib["id"]
                    count += 1
                elif elem.tag == "answer":
                    senses.append(elem.attrib["senseid"])
                    # print elem.attrib["senseid"]
            elif event == "end":
                if elem.tag == "head":
                    word_form = elem.text
                elif elem.tag == "context":
                    context_body = elem.text
                    for child in elem:
                        context_body += " " + child.text + " " + child.tail
                    # Save every 5th as for testing
                    if count == 5:
                        count = 0
                        validation_answers[_id] = (word_pos_pair, set(senses), context_body)

                    else:
                        for feature_type in self.feature_types:
                            nb_model.train(context_body, senses, feature_type, word_form)
                        for sense in senses:
                            nb_model.add_word_sense(sense)
                    senses = []
        accuracy = self.calculate_validation_accuracy(validation_answers)
        print str(accuracy) + "%"

    def calculate_validation_accuracy(self, validation_answers):
        correct = 0
        total = 0
        # baseline = {'bank' : 'bank%1:14:00::',
        #             'begin' : '369204',
        #             'decide' : '1067503',
        #             'degree' : 'degree%1:10:00::',
        #             'difference' : 'difference%1:24:00::',
        #             'different' : 'different%3:00:00::',
        #             'difficulty' : 'difficulty%1:26:00::',
        #             'disc' : 'disc%1:06:01::'}
        for _id in validation_answers:
            word_pos_pair, senses, context_body = validation_answers[_id]
            nb_model = self.naive_bayes_models[word_pos_pair]
            word = word_pos_pair[0]
            collocation_results, prior = nb_model.test(context_body, word, NaiveBayesModel.COLLOCATION)
            coocurrence_results, prior = nb_model.test(context_body, word, NaiveBayesModel.COOCCURRENCE)
            max_sense = "U"
            max_prob = 0
            for sense_id in collocation_results:
                log_prob = collocation_results[sense_id] + coocurrence_results[sense_id]
                prob = math.exp(log_prob)
                if prob > max_prob:
                    max_sense = sense_id
                    max_prob = prob
            if max_sense in senses:
                correct += 1
            total += 1
        print "Validation: Correct = " + str(correct) + ", Total = " + str(total)
        return float(correct) *100 / total



    def test_data_set(self, path_to_test="test-data.data"):
        """
        Streams through XML file and calculates the probabilties for all senses given an instance
        of a word. Writes the file to output after the entire file is parsed.

        :param path_to_test: the path to test data
        :return:
        """
        # Stream the file in case it won't fit in memory
        context = ET.iterparse(path_to_test, events=("start", "end"))
        context = iter(context)
        nb_model = None
        word = None
        _id = None
        results = {}
        prior = {}
        for feature_types in self.feature_types:
            results[feature_types] = {}
        for event, elem in context:
            if event == "start":
                if elem.tag == "lexelt":
                    word, pos = elem.attrib["item"].split(".")
                    word_pos_pair = (word, pos)
                    # Error message for unseen word
                    if word_pos_pair not in self.naive_bayes_models:
                        print "-" * 80
                        print "Encountered word not in dictionary: " + word_pos_pair
                        print "-" * 80
                    # Use trained Naive Bayes model for this word
                    else:
                        nb_model = self.naive_bayes_models[word_pos_pair]
                elif elem.tag == "instance":
                    _id = elem.attrib["id"]
            elif event == "end":
                if elem.tag == "context":
                    context_body = elem.text
                    for feature_type in self.feature_types:
                        results[feature_type][_id], prior[_id] = nb_model.test(context_body, word, feature_type)
        # Write output to file
        self.write_output(results, prior)

    def write_output(self, results, prior):
        """
        For each test instance, writes to an output file the most likely sense for each instance.

        :param results: all conditional probabilities for each feature type
        :param prior: all prior probabilities of a sense
        """
        with open('PSKJ_Learning.csv', 'w') as f:
            f.write("Id,Prediction\n")
            collocation_results = results[NaiveBayesModel.COLLOCATION]
            coocurrence_results = results[NaiveBayesModel.COOCCURRENCE]
            for instance_id in collocation_results:
                # Calculate most likely sense for this instance
                max_sense, max_prob = self.make_prediction(coocurrence_results, collocation_results, instance_id, prior)
                f.write(instance_id + "," + max_sense + "\n")

    def make_prediction(self, coocurrence_results, collocation_results, instance_id, prior):
        """
        Makes a prediction given the instance id of the test point

        :param coocurrence_results: probabilities for coocurrence features
        :param collocation_results: probabilities for collocation features
        :param instance_id: instance id of test
        :param prior: prior probabilities of each sense
        :return max_sense, max_prob: the most like sense and its probability
        """
        sense_values = collocation_results[instance_id]
        max_sense = "U"
        max_prob = 0
        for sense_id in sense_values:
            # Calculate the log probability
            log_prob = sense_values[sense_id] + coocurrence_results[instance_id][sense_id]
            # Calculate the actual probability
            prob = math.exp(log_prob + prior[instance_id][sense_id])
            # Update sense if the new probability is higher than seen before
            if prob > max_prob:
                max_sense = sense_id
                max_prob = prob
        return max_sense, max_prob

class CollocationFeatureVector:
    """
    Representation of a collocation feature. Collocation feature vectors are a list of
    parts of speech tags around a given word.

    Authors: Philip Su ps845, Kelvin Jin kkj9
    """

    def __init__(self, context_body, word):
        """
        Initializes the feature vector as a list with parts of speech tags around the word.

        :param context_body: context of a word
        :param word: word to determine sense of
        """
        self.feature_vector = []
        context_body = context_body.split()
        # Generate list of tagged part of speeches
        pos_body = NaiveBayesModelCatalog.tagger.tag(context_body)
        # Find all indices of word
        word_indices = [i for i in range(len(context_body)) if context_body[i] == word]
        # Generate a vector for every instance of the word in the context
        for index in word_indices:
            # Only use words with enough context
            if (index >= NaiveBayesModel.N_CONTEXT and
                            index + NaiveBayesModel.N_CONTEXT < len(context_body)):
                for pos_word_pair in pos_body[index - NaiveBayesModel.N_CONTEXT : index + NaiveBayesModel.N_CONTEXT + 1]:
                    pos = pos_word_pair[1]
                    if pos in NaiveBayesModelCatalog.pos_map:
                        pos = NaiveBayesModelCatalog.pos_map[pos]
                    self.feature_vector.append(pos)

    def get_feature_probability_given_sense(self, training_data, sense, category_dim):
        """
        Given a set of training collocation feature vectors, calculates the conditional
        probability of a feature vector given a sense. The product of probabilities become small,
        so operations are done as the summation of logarithms.

        :param training_data: a list of training feature vectors
        :param sense: the sense to calculate the conditional probability for
        :param category_dim: the set of all possible values a feature vector can take
        :return: conditional probability given a set of training features and a sense
        """
        prob = 1
        for i in range(0, len(self.feature_vector)):
            numerator = 0.0
            denominator = 0.0
            test_feature = self.feature_vector[i]
            for training_feature_vector, training_sense in training_data:
                if sense == training_sense:
                    denominator += 1.0
                    if i < len(training_feature_vector.feature_vector):
                        train_feature = training_feature_vector.feature_vector[i]
                        if test_feature == train_feature:
                                numerator += 1
            # Calculate the conditional probability with Laplace smoothing
            prob += math.log((numerator + NaiveBayesModel.SMOOTHING_CONSTANT) / (
                denominator + category_dim * NaiveBayesModel.SMOOTHING_CONSTANT))
        return prob

    def get_values(self):
        """
        Returns the feature vector

        :return: CollocationFeatureVector object
        """
        return self.feature_vector


class CooccurrenceFeatureVector:
    """
    Representation of a coocurrence feature. Coocurence feature vectors are a set of tokens
    in the context of a word.

    Authors: Philip Su ps845, Kelvin Jin kkj9
    """

    def __init__(self, context_body, word):
        """
        Initializes the feature vector as a list with parts of speech tags around the word.

        :param context_body: context of a word
        :param word: word to determine sense of
        """
        self.feature_vector = set()
        for context in context_body.split():
            if word == context or context.lower() in NaiveBayesModelCatalog.stop_words:
                continue
            self.feature_vector.add(context)

    def get_feature_probability_given_sense(self, training_data, sense, category_dim):
        """
        Given a set of training collocation feature vectors, calculates the conditional
        probability of a feature vector given a sense. The product of probabilities become small,
        so operations are done as the summation of logarithms.

        :param training_data: a list of training feature vectors
        :param sense: the sense to calculate the conditional probability for
        :param category_dim: the set of all possible values a feature vector can take
        :return: conditional probability given a set of training features and a sense
        """
        prob = 1
        for test_feature in self.feature_vector:
            numerator = 0.0
            denominator = 0.0
            for training_feature_vector, training_sense in training_data:
                if sense == training_sense:
                    denominator += 1.0
                    if test_feature in training_feature_vector.feature_vector:
                        numerator += 1.0
            # Calculate the conditional probability with Laplace smoothing
            prob += math.log((numerator + NaiveBayesModel.SMOOTHING_CONSTANT) / (
                denominator + category_dim * NaiveBayesModel.SMOOTHING_CONSTANT))
        return prob

    def get_values(self):
        """
        Returns the feature vector

        :return: CollocationFeatureVector object
        """
        return self.feature_vector


if __name__ == '__main__':
    catalog = NaiveBayesModelCatalog()
    # catalog.validate_data_set()
    catalog.train_data_set()
    catalog.test_data_set()
