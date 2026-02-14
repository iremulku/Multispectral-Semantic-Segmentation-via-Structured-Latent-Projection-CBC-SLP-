from __future__ import print_function


def Jaccard(y, y_pred, epsilon=1e-8):   
    TP = (y_pred * y).sum(0)
    FP = ((1-y_pred)*y).sum(0)
    FN = ((1-y)*y_pred).sum(0)
    jack = (TP+epsilon) / (TP+FP+FN+epsilon)
    return jack

def Jaccard2(y, y_pred, epsilon=1e-8):  
    if y.sum(0)==0:
        y = 1-y;
        y_pred= 1-y_pred
        
    TP = (y_pred * y).sum(0)
    FP = ((1-y_pred)*y).sum(0)
    FN = ((1-y)*y_pred).sum(0)
    jack = (TP+epsilon) / (TP+FP+FN+epsilon)
    return jack

def JaccardAndF1(y, y_pred, epsilon=1e-8):  
    if y.sum(0)==0:
        y = 1-y;
        y_pred= 1-y_pred
        
    TP = (y_pred * y).sum(0)
    FP = ((1-y_pred)*y).sum(0)
    FN = ((1-y)*y_pred).sum(0)
    jack = (TP+epsilon) / (TP+FP+FN+epsilon)
    
    
    recall = TP / (TP + FN + epsilon)
    prec = TP / (TP + FP + epsilon)
    f1 = 2 * (recall*prec) / (recall+prec+ epsilon);
    
    return f1