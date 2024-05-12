from common.eval import *

model.eval()

if P.mode == 'test_acc':
    from evals import test_classifier

    with torch.no_grad():
        error = test_classifier(P, model, test_loader, 0, logger=None)

elif P.mode == 'test_marginalized_acc':
    from evals import test_classifier

    with torch.no_grad():
        error = test_classifier(P, model, test_loader, 0, marginal=True, logger=None)

elif P.mode in ['ood', 'ood_pre']:
    if P.mode == 'ood':
        from evals import eval_ood_detection
    else:
        from evals.ood_pre import eval_ood_detection
    
    for param in model.parameters():
            param.requires_grad = True
    auroc_dict = eval_ood_detection(P, model, test_loader, ood_test_loader, P.ood_score,
                                    train_loader=train_loader, simclr_aug=simclr_aug)
        # {'one_class_1': {'CSI': 0.728107},
        #  'one_class_2': {'CSI': 0.9557279999999999},
        #  'one_class_3': {'CSI': 0.9823710000000001},
        #  'one_class_4': {'CSI': 0.945057},
        #  'one_class_5': {'CSI': 0.9775539999999999},
        #  'one_class_6': {'CSI': 0.9593869999999998},
        #  'one_class_7': {'CSI': 0.895971},
        #  'one_class_8': {'CSI': 0.7812140000000001},
        #  'one_class_9': {'CSI': 0.812334},
        #  'one_class_mean': {'CSI': 0.8930803333333334}}
    # ood_score="CSI"
    if P.one_class_idx is not None:
        mean_dict = dict()
        # P.ood_score: ['CSI']
        for ood_score in P.ood_score:
            mean = 0
            for ood in auroc_dict.keys():
                mean += auroc_dict[ood][ood_score]
            mean_dict[ood_score] = mean / len(auroc_dict.keys())
        auroc_dict['one_class_mean'] = mean_dict

    bests = []
    for ood in auroc_dict.keys():
        message = ''
        best_auroc = 0
        for ood_score, auroc in auroc_dict[ood].items():
            message += '[%s %s %.4f] ' % (ood, ood_score, auroc)
            if auroc > best_auroc:
                best_auroc = auroc
        message += '[%s %s %.4f] ' % (ood, 'best', best_auroc)
        if P.print_score:
            print(message)
        bests.append(best_auroc)

    bests = map('{:.4f}'.format, bests)
    print('\t'.join(bests))

else:
    raise NotImplementedError()
