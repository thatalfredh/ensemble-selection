from collections import Counter # for update_weights

import numpy as np # for ensemble prediction
from keras.metrics import binary_crossentropy

np.random.seed(42)

def update_weights(ensemble):
    """ 
    Compute the weights assigned to 
    each model in the ensemble based on frequency 
    ensemble: {'models':[], 'weights':{}, 'predictions':{}} 
    """
    occurrences = Counter(ensemble['models'])
    total = sum(occurences.values())
    ensemble['weights'] = [occurrences[model_idx]/total for model_idx in ensemble['models']]
                           
    return ensemble  

# TO DO: modify for multi-class
def compute_loss(ensemble, y_val):
    """ 
    Compute the ensemble loss from weighted ensemble predictions (cached)
    ensemble: <dict> {'models': [], weights':[], 'predictions':{}}
    y_val: validation ground-truth
    """
    preds = sum([np.array(ensemble['predictions'][model_idx]) * ensemble['weights'][model_idx] 
                 for model_idx in ensemble['models']])
    ensemble_loss = binary_crossentropy(y_true=y_val, y_pred=preds).numpy().mean()
    
    return ensemble_loss

def ensemble_selection(models, validation_data, optimize_metric='loss', iterations=10, threshold=0.005):
    """
    Ensemble selection (with replacement) from a library of neural networks
    models: <list> of <Model> object
    validation_data: <tuple> (X_val, y_val)
    optimize_metrics: <str> 'loss' or 'acc'
    iterations: <int> number of iterations for selection
    """
    
    if optimize_metric not in ['loss', 'acc']:
        print("invalid metric: optimize_metric needs to be 'loss' or 'acc'.")
        return
    
    X_val, y_val = validation_data
    
    # model_pool : {model_idx: {'model': model, 'predictions': predictions} }
    model_pool = {}
    # ensemble : {'models': [model indexes], 'predictions':{model_idx:predictions} 'weights': {model_idx: weight}}
    ensemble = {'models':[], 'predictions':{}, 'weights':{}} 
    
    ensemble_loss, ensemble_acc = None, None
    ensemble_losses = []
    
    # 1 - Initialize model library and get best model for specified metric (acc/loss) - predictions are cached
    best_idx, min_loss, max_acc = None, pow(2, 10), 0
    
    for model_idx, model in enumerate(models): # index of model in <list> models, model <Model> object
        predictions = model.predict(X_val)
        model_pool[model_idx] = {'model': model, 'predictions': predictions}
        loss, acc = model.evaluate(X_val, y_val, batch_size=64, verbose=0, use_multiprocessing=False)
        
        if optimize_metric == 'loss':
            if loss < min_loss:
                best_idx, min_loss = model_idx, loss
        else:
            if acc > max_acc:
                best_idx, max_acc = model_idx, acc
    
    # 2 - Initialize ensemble with Best Model
    ensemble['models'].append(best_idx)
    ensemble['predictions'][best_idx] = model_pool[best_idx]['predictions']
    ensemble = update_weights(ensemble)
    
    if optimize_metric == 'loss':
        ensemble_loss = min_loss
    else:
        ensemble_acc = max_acc
    
    # 3 - Begin Ensemble Selection - TO DO: improve efficiency of algo to linear time by ordered selection
    best_idx = 0 # the only model in the ensemble for now
    for i in range(iterations):
        for model_idx, model in model_pool.items():
            
            ensemble['models'].append(model_idx)
            ensemble['predictions'][model_idx]= model_pool[model_idx]['predictions']
            ensemble = update_weights(ensemble)

            # add to ensemble if score improves
            loss = compute_loss(ensemble, y_val)
            if (ensemble_loss - loss)/ensemble_loss > threshold:
                ensemble_loss = loss
                
            else:
                ensemble['models'].pop()
                if model_idx not in ensemble['models']:
                    ensemble['predictions'].pop(model_idx)
                ensemble = update_weights(ensemble)
            
            ensemble_losses.append(loss)
        print(f"Iteration {i+1:3}: Ensemble Loss = {ensemble_loss}")
    
    # 4 - Final Ensemble
    model_ensemble = {'models':None, 'weights':None}
    model_ensemble['models'] = [model_pool[model_idx]['model'] for model_idx in ensemble['models']]
    model_ensemble['weights'] = [ensemble['weights'][model_idx] for model_idx in ensemble['models']]
    
    return model_ensemble, ensemble_losses

def ensemble_prediction(ensemble, X):
    """ 
    Make ensemble (final model) prediction 
    ensemble: <dict> {'models': [], 'weights':[]}
    X: <array> input feature vector for prediction
    """
    s = len(ensemble['models'])
    prediction =  sum([np.array(ensemble['models'][i].predict(X)) * ensemble['weights'][i] for i in range(s)])
    
    return prediction