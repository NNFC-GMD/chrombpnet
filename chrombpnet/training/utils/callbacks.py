import keras

class LossHistory(keras.callbacks.Callback):
    """
    Callbacks to store train, validation loss at the the end of every batch and the end of every epoch.
    You can also track the counts loss and profile loss seperatley using the callbacks provided.
    Safe for Keras' asynchronous batch-callback dispatch: batches may arrive out of order from a thread pool, so the
    losses are kept by batch index and written in batch order at the end of the epoch (after Keras has waited for
    every pending batch callback).
    """

    async_safe = True
    
    def __init__(self,model_output_path_logs_name,to_track):
        self.model_output_path_logs_name=model_output_path_logs_name
        self.to_track=to_track
        self.outf=open(self.model_output_path_logs_name,'w')
        self.outf.write('Epoch\tBatch\t'+'\t'.join(self.to_track)+'\n')
        keras.callbacks.Callback.__init__(self)

    def on_train_begin(self, logs=None):
        self.losses ={}

    def on_epoch_begin(self,epoch, logs=None):
        # {batch index: [value of each trackable]}
        self.losses[epoch]={}
        self.cur_epoch=epoch
        
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.losses[self.cur_epoch][batch]=[logs.get(trackable) for trackable in self.to_track]
        
    def on_epoch_end(self,epoch,logs=None):
        for i in sorted(self.losses[epoch]):
            self.outf.write(str(epoch)+'\t'+str(i))
            for value in self.losses[epoch][i]:
                self.outf.write('\t'+str(value))
            self.outf.write('\n')

        
    def on_train_end(self,logs=None):
        self.outf.close()
