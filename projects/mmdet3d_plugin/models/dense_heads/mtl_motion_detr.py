import logging
import torch
import torch.nn as nn
import torch.nn.functional as F 
from mmcv.runner import BaseModule
from mmdet3d.models import builder
from mmcv.cnn import build_norm_layer
from mmdet3d.models.builder import HEADS, build_loss

from .bev_encoder import BevEncode
from .map_head import BevFeatureSlicer
from mmcv.runner import auto_fp16, force_fp32
from ..deformable_detr_modules import MHAttentionMap, MaskHeadSmallConv, build_MaskHeadSmallConv, build_detr, build_backbone, build_seg_detr, build_deforamble_transformer, build_position_encoding
from ..deformable_detr_utils import inverse_sigmoid
import pdb
from ...datasets.utils.geometry import cumulative_warp_features_reverse
from ...datasets.utils.instance import predict_instance_segmentation_and_trajectories
from ...datasets.utils.warper import FeatureWarper
from ..motion_modules import warp_with_flow
from mmdet3d.core.bbox.coders import build_bbox_coder



@HEADS.register_module()
class MultiTaskHead_Motion_DETR(BaseModule):
    def __init__(
        self,
        init_cfg=None,
        task_enable=None,
        task_weights=None,
        in_channels=64,
        out_channels_det=64,
        out_channels_map=256,
        bev_encode_block="BottleNeck",
        bev_encoder_type="resnet18",
        bev_encode_depth=[2, 2, 2],
        num_channels=None,
        backbone_output_ids=None,
        norm_cfg=dict(type="BN"),
        bev_encoder_fpn_type="lssfpn",
        bbox_coder=None,
        grid_conf=None,
        det_grid_conf=None,
        map_grid_conf=None,
        motion_grid_conf=None,
        out_with_activision=False,
        using_ego=False,
        shared_feature=False,
        cfg_3dod=None,
        cfg_map=None,
        cfg_motion=None,
        #DETR ARGS HERE
        backbone="resnet18",
        position_embedding="sine",
        num_pos_feats=128,
        return_intermediate_dec= True, 
        block_future_prediction= False,
        #in_channels=64,
        hidden_dim=512,
        nheads=8, 
        enc_layers=6, 
        dec_layers=6, 
        dim_feedforward=512, 
        dropout_transformer=0.1, 
        activation="relu",
        num_feature_levels=4,
        dec_n_points=6, 
        enc_n_points=6, 
        num_queries=150,

        #DETR END  ARGS HERE
        n_future=0,
        flow_warp=False,
        temporal_queries_activated=True,
        train_cfg=None,
        test_cfg=None,
        **kwargs,
    ):
        super(MultiTaskHead_Motion_DETR, self).__init__(init_cfg)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range


        self.fp16_enabled = False
        self.task_enable = task_enable
        self.task_weights = task_weights
        self.using_ego = using_ego
        self.logger = logging.getLogger("timelogger")
        if det_grid_conf is None:
            det_grid_conf = grid_conf

        # build task-features
        self.task_names_ordered = ["map", "3dod", "motion"]
        self.taskfeat_encoders = nn.ModuleDict()
        assert bev_encoder_type == "resnet18"
        self.return_intermediate_dec = return_intermediate_dec
        self.block_future_prediction = block_future_prediction
        # if self.return_intermediate_dec:
        #     self.aux_outputs = dec_layers
        # whether to use shared features
        self.shared_feature = shared_feature
        if self.shared_feature:
            self.taskfeat_encoders["shared"] = BevEncode(
                numC_input=in_channels,
                numC_output=out_channels,
                num_channels=num_channels,
                backbone_output_ids=backbone_output_ids,
                num_layer=bev_encode_depth,
                bev_encode_block=bev_encode_block,
                norm_cfg=norm_cfg,
                bev_encoder_fpn_type=bev_encoder_fpn_type,
                out_with_activision=out_with_activision,
            )
        else:
            for task_name in self.task_names_ordered:
                is_enable = task_enable.get(task_name, False)
                if not is_enable:
                    continue
                
                if task_name == "map":
                    out_channels = out_channels_map
                elif task_name == "3dod" or task_name == "motion":
                    out_channels = out_channels_det
                    task_name = "shared"
                    print("Constructing task_feat_encoder with", task_name)
                else:
                    print("standard outchannels 256")
                    out_channels = 256
                
                self.taskfeat_encoders[task_name] = BevEncode(
                    numC_input=in_channels,
                    numC_output=out_channels,
                    num_channels=num_channels,
                    backbone_output_ids=backbone_output_ids,
                    num_layer=bev_encode_depth,
                    bev_encode_block=bev_encode_block,
                    norm_cfg=norm_cfg,
                    bev_encoder_fpn_type=bev_encoder_fpn_type,
                    out_with_activision=out_with_activision,
                )


        
        # build task-decoders
        self.task_decoders = nn.ModuleDict()
        self.task_feat_cropper = nn.ModuleDict()

        # 3D object detection
        self.det_enabled = task_enable.get("3dod", False)
        if task_enable.get("3dod", False):
            cfg_3dod.update(train_cfg=train_cfg)
            cfg_3dod.update(test_cfg=test_cfg)

            self.task_feat_cropper["shared"] = BevFeatureSlicer(
                grid_conf, det_grid_conf)
            self.task_decoders["3dod"] = builder.build_head(cfg_3dod)

        # static map
        self.map_enabled = task_enable.get("map", False)
        if task_enable.get("map", False):
            cfg_map.update(train_cfg=train_cfg)
            cfg_map.update(test_cfg=test_cfg)

            self.task_feat_cropper["map"] = BevFeatureSlicer(
                grid_conf, map_grid_conf)
            self.task_decoders["map"] = builder.build_head(cfg_map)

        # motion_head
        self.motion_enabled =False
        if task_enable.get("motion", False):
            cfg_motion.update(train_cfg=train_cfg)
            cfg_motion.update(test_cfg=test_cfg)
            self.motion_enabled = True
            self.task_feat_cropper["shared"] = BevFeatureSlicer(
                grid_conf, motion_grid_conf
            )
            self.task_decoders["motion"] = builder.build_head(cfg_motion)

        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.num_queries = num_queries
        
        self.backbone = build_backbone(backbone=backbone, layers=[
                   2, 2, 2, 2], return_feature_layers=True, position_embedding=position_embedding, num_pos_feats=num_pos_feats, hidden_dim=hidden_dim, num_feature_level=num_feature_levels)


        self.transformer = build_deforamble_transformer(hidden_dim, nheads, enc_layers, dec_layers,
                                                        dim_feedforward, dropout_transformer, activation,
                                                        num_feature_levels, dec_n_points, enc_n_points,
                                                        num_queries, return_intermediate_dec)
        # self.DETR = build_detr(
        #     self.backbone, self.transformer, num_classes, num_queries, num_feature_levels)
        
        self.temporal_queries_activated = temporal_queries_activated 
            
        self.flow_warp = flow_warp
        self.warper = FeatureWarper(grid_conf=grid_conf)
        self.two_stage = False
        self._init_detr_layers()
        
    def _init_detr_layers(self):
        if self.temporal_queries_activated:
            self.temporal_query_projection = nn.Sequential(
                nn.Linear(self.num_queries, out_features=self.num_queries),
                nn.Dropout(p=0.1),
                nn.ReLU(),
            )
        num_backbone_outs = len(self.backbone.strides)
        input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = self.backbone.num_channels[_]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                nn.GroupNorm(32, self.hidden_dim),
            ))
        for _ in range(self.num_feature_levels - num_backbone_outs):
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, self.hidden_dim,
                            kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, self.hidden_dim),
            ))
            in_channels = self.hidden_dim
        self.input_proj = nn.ModuleList(input_proj_list)
        
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim*2)
        # if self.flow_warp:
        #     self.offset_conv = nn.Sequential(
        #         nn.Conv2d(
        #             in_channels, in_channels, kernel_size=3, padding=1
        #         ),
        #         nn.BatchNorm2d(in_channels),
        #         nn.ReLU(),
        #     )
        #     self.offset_pred = nn.Conv2d(
        #         in_channels, 2, kernel_size=1, padding=0
        #     )
        self.past_query_embed = None 
    
    def _initialize_layers(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)


    def scale_task_losses(self, task_name, task_loss_dict):
        task_sum = 0
        for key, val in task_loss_dict.items():
            task_sum += val.item()
            task_loss_dict[key] = val * self.task_weights.get(task_name, 1.0)

        task_loss_summation = sum(list(task_loss_dict.values()))
        if task_name == "motion":
            cardi = task_loss_dict["cardinality_error"]
            class_error = task_loss_dict["class_error"]
            task_loss_summation = task_loss_summation - cardi - class_error
            if self.return_intermediate_dec and not self.block_future_prediction:
                for i in range(self.aux_outputs):
                    cardi_key = "cardinality_error" + f'_{i+1}'
                    class_key = "class_error" + f'_{i+1}'
                    cardi = task_loss_dict[cardi_key]
                    class_error = task_loss_dict[class_key]
                    task_loss_summation = task_loss_summation - cardi - class_error
                    
        task_loss_dict["{}_sum".format(task_name)] = task_loss_summation

        return task_loss_dict

    def loss(self, predictions, targets):
        loss_dict = {}

        if self.task_enable.get("3dod", False):
            det_loss_dict = self.task_decoders["3dod"].loss(
                gt_bboxes_list=targets["gt_bboxes_3d"],
                gt_labels_list=targets["gt_labels_3d"],
                preds_dicts=predictions["3dod"],
                pc_range=self.pc_range
            )
            loss_dict.update(
                self.scale_task_losses(
                    task_name="3dod", task_loss_dict=det_loss_dict)
            )

        if self.task_enable.get("map", False):
            map_loss_dict = self.task_decoders["map"].loss(
                predictions["map"],
                targets,
            )
            loss_dict.update(
                self.scale_task_losses(
                    task_name="map", task_loss_dict=map_loss_dict)
            )

        if self.task_enable.get("motion", False):
            motion_loss_dict = self.task_decoders["motion"].loss(
                predictions["motion"],targets)
           
            weight_dict = self.task_decoders["motion"].criterion.weight_dict
            for k in motion_loss_dict.keys():
                if k in weight_dict:
                    motion_loss_dict[k] *= weight_dict[k]
                    
            loss_dict.update(
                self.scale_task_losses(
                    task_name="motion", task_loss_dict=motion_loss_dict
                )
            )

        return loss_dict

    def inference(self, predictions, img_metas, rescale):
        res = {}
        print("MTL Head Inference")
        # derive bounding boxes for detection head
        if self.task_enable.get("3dod", False):
            print("MTL Head Inference 3dod")
            res["bbox_list"] = self.get_bboxes(
                predictions["3dod"], img_metas=img_metas, rescale=rescale
            )  # Has len 6 -< and attributes of: reg (2,200,200), height (1,200,200), dim ( (3,200,200)), rot(2,200,200), vel (2,200,200), heatmap (1,200,200)

            # convert predicted boxes in ego to LiDAR coordinates
            if self.using_ego:
                for index, (bboxes, scores, labels) in enumerate(res["bbox_list"]):
                    img_meta = img_metas[index]
                    lidar2ego_rot, lidar2ego_tran = (
                        img_meta["lidar2ego_rots"],
                        img_meta["lidar2ego_trans"],
                    )

                    bboxes = bboxes.to("cpu")
                    bboxes.translate(-lidar2ego_tran)
                    bboxes.rotate(lidar2ego_rot.t().inverse().float())

                    res["bbox_list"][index] = (bboxes, scores, labels)

        # derive semantic maps for map head
        if self.task_enable.get("map", False):
            print("MTL Head Inference map")
            res["pred_semantic_indices"] = self.task_decoders[
                "map"
            ].get_semantic_indices(
                predictions["map"],
            )

        if self.task_enable.get("motion", False):
            print("MTL Head Inference motion")
            seg_prediction, pred_consistent_instance_seg = self.task_decoders[
                "motion"
            ].inference(
                predictions["motion"],
            )

            res["motion_predictions"] = predictions["motion"]
            res["motion_segmentation"] = seg_prediction
            res["motion_instance"] = pred_consistent_instance_seg

        return res

    def forward_with_shared_features(self, bev_feats, targets=None):
        predictions = {}
        auxiliary_features = {}
        print("MTL Head Inference motion")
        bev_feats = self.taskfeat_encoders["shared"]([bev_feats])
        self.logger.debug(
            f"MTL-HEAD forward_with_shared_features bev_feats: {str(bev_feats.shape)}"
        )
        for task_name in self.task_feat_cropper:
            # crop feature before the encoder
            task_feat = self.task_feat_cropper[task_name](bev_feats)
            self.logger.debug(
                f"MTL-HEAD forward_with_shared_features Tasks: {str(task_feat.shape)}"
            )
            # task-specific decoder
            if task_name == "motion":
                task_pred = self.task_decoders[task_name](
                    [task_feat], targets=targets)
            else:
                task_pred = self.task_decoders[task_name]([task_feat])

            predictions[task_name] = task_pred

        return predictions

    def forward(self, bev_feats, targets=None):
        if self.shared_feature:
            return self.forward_with_shared_features(bev_feats, targets)
        # print(
        #     f"Memory allcoated after LLS : {torch.cuda.memory_allocated()/(1<<20):,.0f} MB reserved {torch.cuda.memory_reserved()/(1<<20):,.0f} MB")
        # if bev_feats.isnan().sum() > 0:
        #     print("bev_feats")

        predictions = {}
        #for task_name, task_feat_encoder in self.taskfeat_encoders.items():

        # crop feature before the encoder
        if self.map_enabled:
            task_feat = self.task_feat_cropper["map"](bev_feats)
            self.logger.debug(f"MTL-HEAD foward Tasks: {str(task_feat.shape)}")
            # task-specific feature encoder
            map_feat = self.taskfeat_encoders["map"]([task_feat])
            map_pred = self.task_decoders["map"](
                [map_feat])  # torch.Size([1, 4, 400, 400])
            predictions["map"] = map_pred
            
            
            self.logger.debug(
                f"MTL-HEAD forward Tasks2: {str(task_feat.shape)}")
        
        task_feat = self.task_feat_cropper["shared"](bev_feats)
        #task_feat = self.taskfeat_encoders["shared"]([task_feat])

        # Deformable DETR 
        b,c,h,w = task_feat.shape
        task_mask = mask = torch.zeros(
            (b, h, w), dtype=torch.bool, device=task_feat.device)
        
        features, pos = self.backbone(task_feat, task_mask)
        #features, pos = self.backbone(task_feat, task_mask)
        # for f in features:
        #     for n in f:
        #         if n.isnan().sum() > 0:
        #             print("features")
        # for p in pos:
        #     if p.isnan().sum() > 0:
        #         print("features")
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1][0])  # .tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = task_mask
                mask = F.interpolate(
                    m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](src, mask).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
            if self.temporal_queries_activated: #TODO FIX this 
                if self.current_query is None:
                    self.temporal_query_projection()



        # print(f"Memory allcoated before transformer: {torch.cuda.memory_allocated()/(1<<20):,.0f} MB reserved {torch.cuda.memory_reserved()/(1<<20):,.0f} MB")
        
        # hs, init_reference, inter_references, _, _, seg_memory, seg_mask = self.transformer(
        #     srcs, masks, pos, query_embeds)
        hs, init_reference, inter_references, _, _, seg_memory, seg_mask = self.transformer(srcs, masks, pos, query_embeds)
        # print(
        #     f"Memory allcoated after transformer: {torch.cuda.memory_allocated()/(1<<20):,.0f} MB reserved {torch.cuda.memory_reserved()/(1<<20):,.0f} MB")
        # if hs.isnan().sum() > 0 or hs.sum() == 0.0:
        #     print("hs")
        #dict keys:  'all_cls_scores'  'all_bbox_preds'  'enc_cls_scores' 'enc_bbox_preds'
        if self.det_enabled:
            dod_pred = self.task_decoders["3dod"](
                    hs,init_reference)
            predictions["3dod"] = dod_pred
        

        if self.motion_enabled:
            motion_pred = self.task_decoders["motion"](
                hs, init_reference, seg_memory, seg_mask, features)
            predictions["motion"] = motion_pred
            # print(
            #     f"Memory allcoated after motion head: {torch.cuda.memory_allocated()/(1<<20):,.0f} MB reserved {torch.cuda.memory_reserved()/(1<<20):,.0f} MB")
        
        
        
        
        return predictions

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        future_list = []
        
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
            
        return ret_list
