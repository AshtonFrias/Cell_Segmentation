import torch
import torch.nn.functional as F
import numpy as np

class Metrics:
    @staticmethod
    def m_f1(pred_mask, mask, smooth=1e-10, num_classes=3):
        # Mean F1 score for multiple classes and returns
        with torch.no_grad():
            pred_mask = F.softmax(pred_mask, dim=1)
            pred_mask = torch.argmax(pred_mask, dim=1).contiguous().view(-1)
            mask = mask.contiguous().view(-1)

            f1_per_class = []
            for clas in range(num_classes):
                true_class = pred_mask == clas
                true_label = mask == clas

                if true_label.long().sum().item() == 0:
                    f1_per_class.append(np.nan)
                else:
                    intersect = (true_class & true_label).sum().item()
                    precision = intersect / (true_class.sum().item() + smooth)
                    recall = intersect / (true_label.sum().item() + smooth)

                    f1 = (2 * precision * recall) / (precision + recall + smooth)
                    f1_per_class.append(f1)

            return np.nanmean(f1_per_class)

    @staticmethod
    def pixel_accuracy(output, mask):
        # Accuracy for multiple classes
        with torch.no_grad():
            predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
            correct_predictions = (predictions == mask)
            accuracy = torch.sum(correct_predictions).item() / correct_predictions.numel()
            return accuracy

    @staticmethod
    def acc(pred, target):
        # Accuracy for binary classification
        pred = (pred > 0.5).float()  # Apply threshold (if needed)
        correct = (pred == target).sum()
        total = target.numel()
        accuracy = correct / total
        return accuracy.item()  # Return as a scalar

    @staticmethod
    def f1_score(pred_mask, mask, smooth=1e-10):
        # F1 score for binary classificaiton
        pred_mask = (pred_mask > 0.5).float()  # Threshold probabilities
        mask = mask.float()

        tp = (pred_mask * mask).sum().item()        # True positives
        fp = (pred_mask * (1 - mask)).sum().item()  # False positives
        fn = ((1 - pred_mask) * mask).sum().item()  # False negatives

        precision = tp / (tp + fp + smooth)
        recall = tp / (tp + fn + smooth)
        f1 = 2 * (precision * recall) / (precision + recall + smooth)

        return f1

    '''  IOU FUNCTIONS  '''
    @staticmethod
    def calculate_iou(preds, masks, threshold=0.5):
        # IOU score for binary classificaiton
        preds = torch.sigmoid(preds)  # Apply sigmoid activation
        preds = (preds > threshold).float()  # Convert to binary (0 or 1)

        intersection = (preds * masks).sum()  # TP
        union = preds.sum() + masks.sum() - intersection  # TP + FP + FN

        return (intersection / union).item() if union > 0 else 0  # Avoid division by zero

    @staticmethod
    def m_iou(pred_mask, mask, smooth=1e-10, num_classes=3):
        # Mean IOU function for multiple classes
        with torch.no_grad():
            pred_mask = F.softmax(pred_mask, dim=1)
            pred_mask = torch.argmax(pred_mask, dim=1).contiguous().view(-1)
            mask = mask.contiguous().view(-1)

            iou_per_class = []
            for clas in range(num_classes):
                true_class = pred_mask == clas
                true_label = mask == clas

                if true_label.long().sum().item() == 0:
                    iou_per_class.append(np.nan)
                else:
                    intersect = torch.logical_and(true_class, true_label).sum().item()
                    union = torch.logical_or(true_class, true_label).sum().item()
                    iou = (intersect + smooth) / (union + smooth)
                    iou_per_class.append(iou)

            return np.nanmean(iou_per_class)

    @staticmethod
    def iou_per_class(pred_mask, mask, smooth=1e-10, num_classes=3):
        with torch.no_grad():
            pred_mask = F.softmax(pred_mask, dim=1)
            pred_mask = torch.argmax(pred_mask, dim=1).contiguous().view(-1)
            mask = mask.contiguous().view(-1)   

            iou_per_class = []
            for clas in range(num_classes):
                true_class = pred_mask == clas
                true_label = mask == clas   

                if true_label.long().sum().item() == 0:
                    iou_per_class.append(np.nan)
                else:
                    intersect = torch.logical_and(true_class, true_label).sum().item()
                    union = torch.logical_or(true_class, true_label).sum().item()
                    iou = (intersect + smooth) / (union + smooth)
                    iou_per_class.append(iou)   

            return iou_per_class