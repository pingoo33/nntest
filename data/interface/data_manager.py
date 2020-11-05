import abc


class DataManager(metaclass=abc.ABCMeta):
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.num_adv = 0
        self.advs = []
        self.num_samples = 0

    @abc.abstractmethod
    def load_data(self):
        pass

    @abc.abstractmethod
    def save_advs(self):
        pass

    @abc.abstractmethod
    def mutant_data(self, data):
        pass

    @abc.abstractmethod
    def get_train_data(self):
        pass

    @abc.abstractmethod
    def get_test_data(self):
        pass

    @abc.abstractmethod
    def get_num_samples(self):
        pass

    @abc.abstractmethod
    def get_num_advs(self):
        pass

    @abc.abstractmethod
    def update_sample(self, src_label, dest_label, src=None, dest=None):
        pass

    def reset_sample(self):
        self.num_samples = 0
        self.num_adv = 0
        self.advs = []
