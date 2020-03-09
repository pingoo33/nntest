class StateManager:
    def __init__(self, model_manager, layer_index):
        self.model_manager = model_manager
        self.layer_index = layer_index

    def get_hidden_state(self, data):
        hidden, _, _ = self.model_manager.cal_hidden_state(data, self.layer_index)
        return hidden

    def get_forget_state(self, data):
        _, gate, _ = self.model_manager.cal_hidden_state(data, self.layer_index)
        return gate
