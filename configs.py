class DefaultConfig(object):

    # model name - if provided, will seek to load previous checkpoint and continue training.
    modelname = 'baselineES'

    # file locations
    # srcfile = '/data/NYT/nyt-freebase.train.triples.universal.mention.txt'
    # srcfile = '/data/NYT/nyt-freebase.test.mentions.txt'
    srcfile = '/data/NYT/nyt-freebase.trainandtest.txt'
    datafolder = '/data/train/'
    embedfolder = '/data/glove/'

    # Learning rate
    learning_rate = 1.
    lr_decay = 0.9
    lr_decay_on_generalisation_error = False  # True to decay learning rate when validation error increases

    check_for_early_stop = True

    # Parameter sizes
    embed_size = 100    # Word embeddings.  Use 50, 100, 200 or 300 to match glove embeddings
    hidden_size = 100   # Hidden size for each direction & layer of RNN.
    vocab_size = 10000  # vocab size for long sentences
    vocab_size_short = 1000  # vocab size for short sentences
    rel_vocab_size = 200  # 200 will capture all relations present in training data [185 in all data]

    max_sentence_length = 106   # maximum sequence length for long sentences - needs to be longer than all training data
    max_shortsentence_length = 15   # max seq length for short sentences

    dropout_keep_prob = 1.

    train_size = 0  # 0 to use all remaining data for training.
    validation_size = 4000
    test_size = 8000

    batch_size = 64

    # Reporting & saving frequencies etc.
    report_step = 500
    lr_decay_step = 1000  # for fixed LR decay if not decaying by generalisation error
    save_step = 500
    terminate_step = 20000  # 0 for infinite loop.

    # Weights for each element of cost function
    cost_weight_relation = 1.
    cost_weight_short = 0.


class AssistConfig(DefaultConfig):
    modelname = 'assistedES'
    cost_weight_relation = 1.
    cost_weight_short = 1.


class BaselineDropout(DefaultConfig):
    modelname = 'baselineES_dropout'
    cost_weight_relation = 1.
    cost_weight_short = 0.
    dropout_keep_prob = 0.8


class AssistDropout(DefaultConfig):
    modelname = 'assistedES_dropout'
    cost_weight_relation = 1.
    cost_weight_short = 1.
    dropout_keep_prob = 0.8


class AssistDropout2(AssistDropout):
    modelname = 'assistedES_dropout2'


class TryRegulariastion(DefaultConfig):
    modelname = 'tryregularisationBaseline'


class MixConfig(DefaultConfig):
    def __init__(self, assistfactor):
        self.cost_weight_relation = 1. - assistfactor/10.
        self.cost_weight_short = assistfactor/10.
        self.modelname = self.modelname + str(assistfactor)


class MixConfigNoDropout(DefaultConfig):
    def __init__(self, assistfactor):
        self.cost_weight_relation = 1. - assistfactor/10.
        self.cost_weight_short = assistfactor/10.
        self.dropout_keep_prob = 1.
        self.modelname = 'mixmodel' + str(assistfactor)

class MixConfigDropout(DefaultConfig):
    def __init__(self, assistfactor):
        self.cost_weight_relation = 1. - assistfactor/10.
        self.cost_weight_short = assistfactor/10.
        self.dropout_keep_prob = 0.8
        self.modelname = 'mixmodel_dropout' + str(assistfactor)
