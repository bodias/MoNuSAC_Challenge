from tensorflow.keras.metrics import Recall, TrueNegatives, FalsePositives, Accuracy
from sklearn.metrics import jaccard_score
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt
import numpy as np

#from https://www.kaggle.com/code/yerramvarun/understanding-dice-coefficient
def DICE_COE(mask1, mask2):
    intersect = np.sum(mask1*mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect ) / (fsum + ssum)
    dice = np.mean(dice)
    dice = round(dice, 3) # for easy reading
    return dice


def calculate_metrics(y, y_pred):
    predicted_masks = np.argmax(y_pred, axis=-1)

    m = TrueNegatives()
    m.update_state(y, predicted_masks)
    TN = m.result().numpy()


    m = FalsePositives()
    m.update_state(y, predicted_masks)
    FP = m.result().numpy()

    specificity = TN/(TN+FP)
    print(f"Specificity: {specificity}")

    m = Recall()
    m.update_state(y, predicted_masks)
    sensitivity = m.result().numpy()
    print(f"Sensitivity(Recall): {sensitivity}")

    jaccard = jaccard_score(y.ravel(), predicted_masks.ravel())
    print(f"Jaccard score: {jaccard}")

    dice = DICE_COE(y, predicted_masks)
    print(f"Dice coefficient: {dice}")

    m = Accuracy()
    m.update_state(y, predicted_masks)
    Acc = m.result().numpy()
    print(f"Accuracy: {Acc}")

    ground_truth_labels = y.ravel() 
    score_value = y_pred[:,:,:,1].ravel() #score of the positive class (foreground)
    fpr, tpr, _ = roc_curve(ground_truth_labels,score_value)
    roc_auc = auc(fpr,tpr)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(loc="lower right")