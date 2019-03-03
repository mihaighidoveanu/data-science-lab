
class Model:

    def __init__(self, model):
        self.model = model

    def use_model(self, train, test):
        self.model.fit(train[0], train[1])
        train_score = self.model.score(train[0], train[1])
        test_score = self.model.score(test[0], test[1])
        return train_score, test_score


