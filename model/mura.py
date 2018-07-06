# https://github.com/desimone/Musculoskeletal-Radiographs-Abnormality-Classifier/blob/master/mura.py

import re
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score)


class Mura(object):
    """`
    MURA <https://stanfordmlgroup.github.io/projects/mura/>`_ Dataset :
    Towards Radiologist-Level Abnormality Detection in Musculoskeletal Radiographs.
    """
    _patient_re = re.compile(r'patient(\d+)')
    _study_re = re.compile(r'study(\d+)')
    _image_re = re.compile(r'image(\d+)')
    _study_type_re = re.compile(r'_(\w+)_patient')

    def __init__(self, image_file_names, y_true, y_pred=None):
        self.imgs = image_file_names
        df_img = pd.Series(np.array(image_file_names), name='img')
        self.y_true = y_true
        df_true = pd.Series(np.array(y_true), name='y_true')
        self.y_pred = y_pred
        # number of unique classes
        self.patient = []
        self.study = []
        self.study_type = []
        self.image_num = []
        self.encounter = []
        for img in image_file_names:
            self.patient.append(self._parse_patient(img))
            self.study.append(self._parse_study(img))
            self.image_num.append(self._parse_image(img))
            self.study_type.append(self._parse_study_type(img))
            self.encounter.append("{}_{}_{}".format(
                self._parse_study_type(img),
                self._parse_patient(img),
                self._parse_study(img), ))

        self.classes = np.unique(self.y_true)
        df_patient = pd.Series(np.array(self.patient), name='patient')
        df_study = pd.Series(np.array(self.study), name='study')
        df_image_num = pd.Series(np.array(self.image_num), name='image_num')
        df_study_type = pd.Series(np.array(self.study_type), name='study_type')
        df_encounter = pd.Series(np.array(self.encounter), name='encounter')

        self.data = pd.concat(
            [
                df_img,
                df_encounter,
                df_true,
                df_patient,
                df_patient,
                df_study,
                df_image_num,
                df_study_type,
            ], axis=1)

        if self.y_pred is not None:
            self.y_pred_probability = self.y_pred.flatten()
            self.y_pred = self.y_pred_probability.round().astype(int)
            df_y_pred = pd.Series(self.y_pred, name='y_pred')
            df_y_pred_probability = pd.Series(self.y_pred_probability, name='y_pred_probs')
            self.data = pd.concat((self.data, df_y_pred, df_y_pred_probability), axis=1)

    def __len__(self):
        return len(self.imgs)

    def _parse_patient(self, img_filename):
        return int(self._patient_re.search(img_filename).group(1))

    def _parse_study(self, img_filename):
        return int(self._study_re.search(img_filename).group(1))

    def _parse_image(self, img_filename):
        return int(self._image_re.search(img_filename).group(1))

    def _parse_study_type(self, img_filename):
        return self._study_type_re.search(img_filename).group(1)

    def metrics(self):
        metrics_by_image = {}
        metrics_by_image['accuracy_score'] = accuracy_score(self.y_true, self.y_pred)
        metrics_by_image['f1_score'] = f1_score(self.y_true, self.y_pred)
        metrics_by_image['precision_score'] = precision_score(self.y_true, self.y_pred)
        metrics_by_image['recall_score'] = recall_score(self.y_true, self.y_pred)
        metrics_by_image['cohen_kappa_score'] = cohen_kappa_score(self.y_true, self.y_pred)
        return metrics_by_image

    def metrics_by_encounter(self):
        y_pred = self.data.groupby(['encounter'])['y_pred_probs'].mean().round()
        y_true = self.data.groupby(['encounter'])['y_true'].mean().round()

        encounter_metrics['accuracy_score'] = accuracy_score(y_true, y_pred)
        encounter_metrics['f1_score'] = f1_score(y_true, y_pred)
        encounter_metrics['precision_score'] = precision_score(y_true, y_pred)
        encounter_metrics['recall_score'] = recall_score(y_true, y_pred)
        encounter_metrics['cohen_kappa_score'] = cohen_kappa_score(self.y_true, self.y_pred)

        return encounter_metrics

    def metrics_by_study_type(self):
        y_pred = self.data.groupby(['study_type', 'encounter'])['y_pred_probs'].mean().round()
        y_true = self.data.groupby(['study_type', 'encounter'])['y_true'].mean().round()

        study_type_metrics['accuracy_score'] = accuracy_score(y_true, y_pred)
        study_type_metrics['f1_score'] = f1_score(y_true, y_pred)
        study_type_metrics['precision_score'] = precision_score(y_true, y_pred)
        study_type_metrics['recall_score'] = recall_score(y_true, y_pred)
        study_type_metrics['cohen_kappa_score'] = cohen_kappa_score(self.y_true, self.y_pred)

        return study_type_metrics


