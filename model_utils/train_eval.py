import torch

from utils.mura_utils import load_checkpoint
from metrics.evaluation import metrics_by_value


PRINT_EACH = 400
BATCH_SIZE = 64


def train(model, num_epoch, train_dataloader, eval_dataloader,
          optimizer, criterion, save_path='./model_checkpoints/', pretrained=False):
    """Train model

    Arguments
        model :

        num_epoch :

        train_dataloader :

        eval_dataloader :

        optimizer :

        criterion :

        save_path :

        pretrained :

    """
    if pretrained:
        load_checkpoint(model, optimizer)

    model.train()
    running_loss = 0.0
    metrics_prev = 0

    for epoch in range(num_epoch):
        print('EPOCH-----{}'.format(epoch))
        for batch_idx, data in enumerate(train_dataloader, 0):
            # get the inputs
            X_train, y_train, _, _ = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            print('Current loss is -------- {}'.format(loss.item()))
            running_loss += loss.item()
            if batch_idx % PRINT_EACH == 0:
                print('\n[{}, {}]'.format(epoch + 1, batch_idx + 1))
                print(' Train loss: {}\n'.format(running_loss / PRINT_EACH))

        metrics_by_image, _, _ = test(model, eval_dataloader, verbose=1)

        # save best params
        if metrics_by_image['f1_score'] > metrics_prev:
            # save_checkpoint(model, optimizer)
            metrics_prev = metrics_by_image['f1_score']


def test(model, eval_dataloader, verbose=None):
    """Evaluate model

    Use trained model for evaluation with default metrics
    and custom metrics from the MURA research paper 1712.06957.pdf.

    Arguments
        model :

        eval_dataloader :

    Returns

    """
    model = model.eval() # set evaluation mode
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(eval_dataloader, 0):
            X_val, y_val, patient_val, study_type_val = data

            y_pred = model(X_val)
            values, indices = torch.max(y_pred, 1)
            y_pred = indices
            image_metrics = metrics_by_value(y_val, y_pred)
            metrics_by_patient = metrics_by_value(y_val, y_pred, patient_val)
            metrics_by_study_type = metrics_by_value(y_val, y_pred, study_type_val)

            if verbose is not None:
                print("\nMETRICA BY IMAGE")
                print("ACURACCY SCORE----------- {}".format(image_metrics['accuracy_score']))
                print("F1 SCORE----------------- {}".format(image_metrics['f1_score']))
                print("PRECISION SCORE---------- {}".format(image_metrics['precision_score']))
                print("RECALL SCORE------------- {}\n".format(image_metrics['recall_score']))

                print("\nMETRICA BY PATIENT")
                print("ACURACCY SCORE----------- {}".format(metrics_by_patient['accuracy_score']))
                print("F1 SCORE----------------- {}".format(metrics_by_patient['f1_score']))
                print("PRECISION SCORE---------- {}".format(metrics_by_patient['precision_score']))
                print("RECALL SCORE------------- {}\n".format(metrics_by_patient['recall_score']))

                print("\nMETRICA BY STUDY TYPE")
                print("ACURACCY SCORE----------- {}".format(metrics_by_study_type['accuracy_score']))
                print("F1 SCORE----------------- {}".format(metrics_by_study_type['f1_score']))
                print("PRECISION SCORE---------- {}".format(metrics_by_study_type['precision_score']))
                print("RECALL SCORE------------- {}\n\n\n".format(metrics_by_study_type['recall_score']))

    return image_metrics, metrics_by_patient, metrics_by_study_type

