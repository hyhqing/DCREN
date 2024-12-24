# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
        # miou
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        # mean acc
        acc = np.diag(self.hist).sum() / self.hist.sum()
        # Recall, F1
        recall = np.diag(self.hist) / self.hist.sum(axis=1)
        f1 = 2 * ((np.diag(self.hist) / self.hist.sum(axis=0)) * recall) / ((np.diag(self.hist) / self.hist.sum(axis=0))+ recall)
        # Calculate Kappa
        p0 = np.diag(self.hist).sum() / self.hist.sum()
        pe = 0
        for i in range(len(self.hist)):
            pe += np.sum(self.hist[i]) * np.sum(self.hist[:, i])
        pe = pe / np.sum(self.hist) ** 2
        kappa = (p0 - pe) / (1 - pe)
        # Calculate FNR
        fn = np.sum(self.hist, axis=1) - np.diag(self.hist)  # False Negatives
        fp = np.sum(self.hist, axis=0) - np.diag(self.hist)  # False Positives
        tp = np.diag(self.hist)  # True Positives
        tn = np.sum(self.hist) - (fp + fn + tp)  # True Negatives

        fnr = fn / (fn + tp)  # False Negative Rate

        return acc, recall, f1, iou, miou, kappa, fnr



if __name__ == '__main__':
    label_path = r'C:\Users\admin\Desktop\DCREN\run/'
    predict_path = r'C:\Users\admin\Desktop\DCREN\run/'
    pres = os.listdir(predict_path)
    labels = []
    predicts = []
    # # 创建一个IOUMetric实例
    # iou_calculator = IOUMetric(num_classes=2)  # 假设您有两个类别
    for im in pres:
        if im[-4:] == '.png':
            label_name = im.split('.')[0] + '.png'
            lab_path = os.path.join(label_path, label_name)
            pre_path = os.path.join(predict_path, im)
            print(pre_path)
            label = cv2.imread(lab_path,0)
            pre = cv2.imread(pre_path,0)
            label[label>0] = 1
            pre[pre>0] = 1
            labels.append(label)
            predicts.append(pre)
    el = IOUMetric(2)
    acc, recall, f1, iou, miou, kappa, fnr = el.evaluate(predicts, labels)
    print('acc: ', acc)
    print('recall: ', recall)
    print('f1: ', f1)
    print('iou: ', iou)
    print('miou: ', miou)
    print('kappa: ', kappa)
    print('fnr: ', fnr)
