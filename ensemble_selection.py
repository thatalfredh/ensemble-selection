# ensemble usage
from collections import Counter

def update_ensemble_weights(ensemble):
    
    """ This function computes the weights assigned to each model in the ensemble based on frequency """
    
    #from collections import Counter
    
    occurrences = Counter(ensemble['models'])
    nb_samples = sum([cnt for model_idx, cnt in occurrences.items()])
    
    for model_idx in ensemble['models']:
        ensemble['weights'][model_idx] = occurrences[model_idx]/nb_samples
        
    return ensemble    

def compute_ensemble_loss(ensemble, y_val):
    
    """ This function computes the ensemble loss from weighted ensemble predictions """
    
    #import numpy as np
    #from keras.metrics import binary_crossentropy
    
    predictions = 0
    for model_idx in ensemble['models']:
#         print(np.array(ensemble['predictions'][model_idx])) 
#         print(ensemble['weights'][model_idx])
        predictions += np.array(ensemble['predictions'][model_idx]) * ensemble['weights'][model_idx]
    
    ensemble_loss = binary_crossentropy(y_true=y_val, y_pred=predictions).numpy().mean()
    
    return ensemble_loss

def ensemble_selection(models, validation_data, optimize_metric='loss', iterations=15, threshold=0.001):
    """
    This function implements ensemble selection (with replacement) from a library of neural networks
    Parameters:
    models: <list> keras models
    validation_data: <tuple> (X_val, y_val)
    optimize_metrics: <str> 'loss', 'acc'
    iterations: <int> number of iterations for selection
    """
    
    #from copy import deepcopy
    #import numpy as np
    #np.random.seed(42)
    #from keras.metrics import binary_crossentropy
    
    if optimize_metric not in ['loss', 'acc']:
        print("optimize_metric needs to be 'loss' or 'acc'.")
        return
    
    X_val, y_val = validation_data
    # model pool {model_idx: {'model': model, 'predictions': predictions} }
    model_pool = {} 
    # ensemble {'models': [model indexes], 'predictions':{model_idx:predictions} 'weights': {model_idx: weight}}
    ensemble = {'models':[], 'predictions':{}, 'weights':{}} 
    ensemble_loss = pow(2,10)
    ensemble_losses = []
    
    # initialize model library and get best model
    best_idx, min_loss = 0, pow(2, 10)
    for model_idx, model in enumerate(models):
        predictions = model.predict(X_val)
        model_pool[model_idx] = {'model': model, 'predictions': predictions}
        
        loss, acc = model.evaluate(X_val, y_val)
        if optimize_metric == 'loss':
            if binary_crossentropy(y_val, predictions).numpy().mean() < min_loss:
                best_idx = model_idx
        else:
            # TODO: implement for accuracy
            pass
    
    # initialize ensemble
    ensemble['models'].append(best_idx)
    ensemble['predictions'][best_idx]= model_pool[best_idx]['predictions']
    ensemble = update_ensemble_weights(ensemble)
    #print(ensemble)
    if optimize_metric == 'loss':
        ensemble_loss = binary_crossentropy(y_val, model_pool[best_idx]['predictions']).numpy().mean()
    else:
        #TODO: implement for accuracy
        pass
    
    # ensemble selections
    best_idx = 0
    for i in range(iterations):
        for model_idx, model in model_pool.items():
            # try to add
            ensemble_copy = deepcopy(ensemble)
            ensemble_copy['models'].append(model_idx)
            ensemble_copy['predictions'][model_idx]= model_pool[best_idx]['predictions']
            ensemble_copy = update_ensemble_weights(ensemble_copy)

            # add to ensemble if score improves
            loss = compute_ensemble_loss(ensemble_copy, y_val)
            #print(loss)
            
            #print(ensemble_loss)
            if (ensemble_loss - loss)/ensemble_loss > threshold:
                ensemble_loss = loss
                ensemble = deepcopy(ensemble_copy)
                ensemble_losses.append(loss)
            
        print(f"Iteration {i+1:3}: Ensemble Loss = {ensemble_loss}")  
        #print(ensemble)
        #print()
        
    model_ensemble = {'models':[], 'weights':[]}
    for model_idx in ensemble['models']:
        model_ensemble['models'].append(model_pool[model_idx]['model'])
        model_ensemble['weights'].append(ensemble['weights'][model_idx])
        
    return model_ensemble, ensemble_losses

def ensemble_prediction(model_ensemble, X):
    """ 
    This function makes ensemble prediction 
    model_ensemble: <dict> {'models': [], 'weights':[]}
    X: feature vector for prediction
    """
    predictions = 0
    for model_idx in range(len(model_ensemble['models'])):
        predictions += np.array(model_ensemble['models'][model_idx].predict(X)) * model_ensemble['weights'][model_idx]
        
    return predictions