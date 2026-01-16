# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 


from __future__ import division

import glob
import os.path as osp

import numpy as np
import onnxruntime
from numpy.linalg import norm

from ..model_zoo import model_zoo
from ..utils import DEFAULT_MP_NAME, ensure_available_local
from .common import Face

__all__ = ['FaceAnalysis']

class FaceAnalysis:
    def __init__(self, name=DEFAULT_MP_NAME, root='~/.insightface', allowed_modules=None, **kwargs):
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        self.model_dir = root
        onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def detect(self,img, max_num=0, det_metric='default'):
        bboxes, _ = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric=det_metric)
        return bboxes.shape[0]


    def get(self, img, max_num=0, det_metric='default'):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric=det_metric)
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname=='detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return ret

    def draw_on(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            # 1. 获取坐标和置信度
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box
            score = face.det_score if hasattr(face, 'det_score') else 0.0

            # 定义颜色 (BGR 格式)
            theme_color = (0, 255, 0)  # 亮绿色
            txt_color = (255, 255, 255) # 白色

            cv2.rectangle(dimg, (x1, y1), (x2, y2), theme_color, 3)

            label = f'Conf: {score:.2f}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (t_w, t_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            cv2.rectangle(dimg, (x1, y1 - t_h - 10), (x1 + t_w + 5, y1), theme_color, -1)
            cv2.putText(dimg, label, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            # 4. 绘制 5 点关键点 (画大并使用实心)
            if hasattr(face, 'kps') and face.kps is not None:
                for (kx, ky) in face.kps:
                    # 画一个带黑边的大圆点，增加辨识度
                    cv2.circle(dimg, (int(kx), int(ky)), 4, (0, 0, 0), -1)      # 黑底
                    cv2.circle(dimg, (int(kx), int(ky)), 2, (0, 255, 255), -1) # 黄心

        return dimg

