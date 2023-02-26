# Copyright (c) OpenMMLab. All rights reserved.
from .map_head import MapHead
from .mtl_head import MultiTaskHead
from .motion_head import MotionHead
from .det_head import CenterHeadv1

from .dgcnn3d_head import DGCNN3DHead
from .detr3d_head import Detr3DHead
from .petr_head import PETRHead
from .petrv2_head import PETRv2Head
from .petr_head_seg import PETRHeadseg
from .petrv2_dnhead import PETRv2DNHead
from .mtl_motion_detr import MultiTaskHead_Motion_DETR
from .motion_detr_det import Motion_DETR_DET


__all__ = ['MapHead', 'MultiTaskHead', 'MotionHead', 'CenterHeadv1', 'DGCNN3DHead',
           'Detr3DHead', 'PETRHead', 'PETRv2Head', 'PETRHeadseg', 'PETRv2DNHead', 'MultiTaskHead_Motion_DETR', 'Motion_DETR_DET']
