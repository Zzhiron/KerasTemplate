from Base import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, train_data, val_data, model, config):
        super(Trainer, self).__init__(train_data, val_data, model, config)
        
    def train(self):
        History = self.model.fit(
            self.train_data,
            steps_per_epoch=self.train_steps,
            epochs=self.epochs,
            callbacks=self.callbacks_list,
            validation_data=self.val_data,
            verbose=1
        )
