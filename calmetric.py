import os,sys
import numpy as np
def stastics(resultpth):
    lstdir = os.listdir(resultpth)
    max_acc = []
    max_auc = []
    auc_max = []
    acc_max = []
    recall = []
    precision = []
    specificity = []
    f1 = []
    for i in lstdir:
        result = np.load(resultpth+i,allow_pickle=True).tolist()
                                                                #print(result)
                                                                #         for j in result['max_dict']:
                                                                #             if result['max_dict'][j][1] == 0 and result['max_dict'][j][2] == 0:
                                                                #                 print(j,result['max_dict'][j])
        print(i,'max_acc',result['max_acc'],'max_auc',result['max_auc'],'acc_max',result['acc_max'],'auc_max',result['auc_max'],'recall',result['recall'],'prec',result['prec'],'spec',result['spec'],'f1score',result['f1score'])
        #print('data',result['max_auc_dict'])
        max_acc.append(result['max_acc'])
        max_auc.append(result['max_auc'])
        acc_max.append(result['acc_max'])
        auc_max.append(result['auc_max'])
        recall.append(result['recall'])
        precision.append(result['prec'])
        specificity.append(result['spec'])
        f1.append(result['f1score'])
    mean_acc= np.mean(max_acc)
    std_acc = np.std(max_acc)
    print('max_acc',mean_acc,std_acc)
    mean_auc= np.mean(max_auc)
    std_auc = np.std(max_auc)
    print('max_auc',mean_auc,std_auc)
    mean_acc= np.mean(acc_max)
    std_acc = np.std(acc_max)
    print('acc_max',mean_acc,std_acc)
    mean_auc= np.mean(auc_max)
    std_auc = np.std(auc_max)
    print('auc_max',mean_auc,std_auc)
    mean_recall= np.mean(recall)
    std_recall = np.std(recall)
    print('recall',mean_recall,std_recall)
    mean_prec = np.mean(precision)
    std_prec = np.std(precision)
    print('prec',mean_prec,std_prec)
    mean_spec = np.mean(specificity)
    std_spec = np.std(specificity)
    print('spec',mean_spec,std_spec)
    mean_f1 = np.mean(f1)
    std_f1 = np.std(f1)
    print('f1',mean_f1,std_f1)

if __name__ == '__main__':
    path = sys.argv[1]
    print(path)
    stastics(path)
