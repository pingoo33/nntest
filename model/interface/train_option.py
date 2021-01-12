from tensorflow.keras.optimizers import Optimizer


class TrainOption:
    def __init__(self, epochs, batch_size, opt: Optimizer, loss: str, metrics):
        self.epochs = epochs
        self.batch_size = batch_size
        self.opt = opt
        self.loss = loss
        self.metrics = metrics
