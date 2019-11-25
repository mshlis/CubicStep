import numpy as np
import tensorflow as tf
import time

def train(model, dataset, optimizer, loss_fn, epochs=10, optimizer_use_loss=True, verbose=0):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            prediction = model(x)
            loss = loss_fn(y, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
        if optimizer_use_loss:
            optimizer.apply_gradients(loss, zip(gradients, model.trainable_variables))
        else:
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    losses = []
    for ep in range(epochs):
        loss_x = 0.
        t0 = time.time()
        for i, (x, y) in enumerate(dataset):
            loss = train_step(x, y)
            loss_x = loss_x + loss
            if np.isnan(loss.numpy()): return losses
            #print(f'\tstep {i}: {loss.numpy()}')
        tnet = time.time() - t0
        losses.append((loss_x / (i+1)).numpy())
        if verbose: print(f'epoch {ep}: {losses[-1]} in {tnet:.2f} seconds')
    return np.array(losses)