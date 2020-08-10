import numpy as np

class Learner:
    
    def __init__(self, loss, model, opt, epochs):
        self.loss = loss
        self.model = model
        self.opt = opt
        self.epochs = epochs
        self.cbs = []
        
    def set_callbacks(self, cblist):
        for cb in cblist:
            self.cbs.append(cb)
            
    def __call__(self, cbname, *args):
        status = True
        for cb in self.cbs:
            cbwanted = getattr(cb, cbname, None)
            status = status and cbwanted and cbwanted(*args)
        return status
    
    def train_loop(self, dl, X_train,y_train, X_test,y_test):
        self.dl = dl # dl added in here
        bs = self.dl.bs
        datalen = len(self.dl.data)
        self.bpe = datalen//bs
        self.afrac = 0.
        if datalen % bs > 0:
            self.bpe  += 1
            self.afrac = (datalen % bs)/bs
        self('fit_start')
        for epoch in range(self.epochs):
            self('epoch_start', epoch)
            for inputs, targets in dl:
                self("batch_start", dl.current_batch)
                
                # make predictions
                predicted = self.model(inputs)

                # actual loss value
                epochloss = self.loss(predicted, targets)
                self('after_loss', epochloss)

                # calculate gradient
                intermed = self.loss.backward(predicted, targets)
                self.model.backward(intermed)

                # make step
                self.opt.step(self.model)
                
                self('batch_end')
            #New code to find training and testing accuracy
            pred_tr= self.model(X_train)
            pred_tr= 1*(pred_tr>=0.5)
            acc=(np.sum(1*(pred_tr==y_train.reshape(-1,1)))/len(y_train))*100
            
            pred_te= self.model(X_test)
            pred_te= 1*(pred_te>=0.5)
            acc_test=(np.sum(1*(pred_te==y_test.reshape(-1,1)))/len(y_test))*100
            
            #The accuracies are passed to epoch_end
            self('epoch_end', acc, acc_test)
        self('fit_end')
        return epochloss