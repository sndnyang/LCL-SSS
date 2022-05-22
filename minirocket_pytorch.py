from tsai.basics import *
import sktime
import sklearn

from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *

from scipy.io import loadmat
from data_utils import get_data
from matplotlib import pyplot as plt


if __name__ == '__main__':
    train_data = loadmat('earthb.mat')
    all_data = train_data['images']
    all_target = train_data['labels']

    # load data
    x_train, x_valid, x_test, y_train, y_valid, y_test, splits, splits_test = get_data(all_data, all_target, dataset='eq', seed=1)
    # set contains training and validation
    X = np.concatenate([x_train, x_valid])
    y = np.concatenate([y_train, y_valid])

    training = True
    if len(sys.argv) > 1:
        training = False
    print("train or evaluate? ", 'Train!' if training is True else 'Evaluate!')
    if training is True:
        mrf = MiniRocketFeatures(x_train.shape[1], x_train.shape[2]).to(default_device())
        mrf.fit(x_train)

        X_feat = get_minirocket_features(X, mrf, chunksize=512, to_np=True)

        # As above, use tsai to bring X_feat into fastai, and train.
        tfms = [None, TSClassification()]
        batch_tfms = TSStandardize(by_sample=True)
        dls = get_ts_dls(X_feat, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)

        model = build_ts_model(MiniRocketHead, dls=dls)
        learn = Learner(dls, model, metrics=accuracy)

        timer.start()
        learn.fit_one_cycle(30, 3e-4)    # epoch 30 (20 ~ 100),  learning rate 3e-4
        timer.stop()

        new_feat = get_minirocket_features(X[splits[1]], mrf, chunksize=1024, to_np=True)
        probas, _, pred = learn.get_X_preds(new_feat)
        print('Valid Accuracy', sklearn.metrics.accuracy_score(y[splits[1]], pred.astype(int)))

        new_feat = get_minirocket_features(x_test, mrf, chunksize=1024, to_np=True)
        probas, _, pred = learn.get_X_preds(new_feat)
        pred = pred.astype(int)
        print('Test accuracy', sklearn.metrics.accuracy_score(y_test, pred))

        PATH = Path("./models/MR_feature.pt")
        PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mrf.state_dict(), PATH)
        PATH = Path('./models/MR_learner.pkl')
        learn.export(PATH)
        print('model save path', PATH)
    else:
        mrf = MiniRocketFeatures(X.shape[1], X.shape[2]).to(default_device())
        PATH = Path("./models/MR_feature.pt")
        mrf.load_state_dict(torch.load(PATH))
        PATH = Path('./models/MR_learner.pkl')
        learn = load_learner(PATH, cpu=False)
        new_feat = get_minirocket_features(x_test, mrf, chunksize=1024, to_np=True)
        probas, _, pred = learn.get_X_preds(new_feat)
        pred = pred.astype(int)
        print('Test accuracy', sklearn.metrics.accuracy_score(y_test, pred))

    from sklearn.metrics import precision_recall_fscore_support
    print('precision recall  F1 score in micro:', precision_recall_fscore_support(y_test, pred, average='micro'))
    print('precision recall  F1 score in macro:', precision_recall_fscore_support(y_test, pred, average='macro'))

    from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    disp.plot()
    plt.savefig('resnet_confusion_matrix.png')
    plt.close()
