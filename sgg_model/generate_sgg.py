# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from models import build_model
import os
import json

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    parser.add_argument('--img_path', type=str, default='demo/vg1.jpg',
                        help="Path of the test image")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='ckpt/checkpoint0149.pth', help='resume from checkpoint')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")


    # distributed training parameters
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser



# %cd RelTR/
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
import requests
import matplotlib.pyplot as plt

"""# VG labels
VG 150 enitiy classes and 50 relationship classes.
"""

CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

"""# Build and load the pretrained model"""

from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine
from models.transformer import Transformer
from models.reltr import RelTR

position_embedding = PositionEmbeddingSine(128, normalize=True)
backbone = Backbone('resnet50', False, False, False)
backbone = Joiner(backbone, position_embedding)
backbone.num_channels = 2048

transformer = Transformer(d_model=256, dropout=0.1, nhead=8,
                          dim_feedforward=2048,
                          num_encoder_layers=6,
                          num_decoder_layers=6,
                          normalize_before=False,
                          return_intermediate_dec=True)

model = RelTR(backbone, transformer, num_classes=151, num_rel_classes = 51,
              num_entities=100, num_triplets=200)

# Some transformation functions
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
          (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return b

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    resized = [b[0]*img_w, b[1]*img_h, b[2]*img_w, b[3]*img_h]
    return resized

def predict_sgg(img_path, ckpt_path):
    """# Load Image
    You can replace the link with other images. Note that the entities in the used image should be included in the VG labels.
    """

    im = Image.open(img_path)
    plt.imshow(im)
    img = transform(im).unsqueeze(0)

    top_vals = 50

    """# Inference"""

    # The checkpoint is pretrained on Visual Genome
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # propagate through the model
    outputs = model(img)

    # keep only predictions with >0.3 confidence
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1][:top_vals]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1][:top_vals]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1][:top_vals]
    keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                            probas_obj.max(-1).values > 0.3))

    kg_metadata_dict = {'bbox': [], 'size': im.size, 'mode': 'xyxy', 'extra_fields':{'pred_labels': [], 'pred_scores': [], 'rel_pair_idxs': [], 'pred_rel_scores':[], 'pred_rel_labels':[]}, "triplet_extra_fields": []}

    for i in range(0, len(probas)):
        kg_metadata_dict['bbox'].append(rescale_bboxes(outputs['sub_boxes'][0][i].tolist(), im.size))
        kg_metadata_dict['extra_fields']['pred_labels'].append(probas_sub[i].argmax().item())
        kg_metadata_dict['extra_fields']['pred_scores'].append(probas_sub[i][probas_sub[i].argmax()].item())

    
    for i in range(0, len(probas)):
        kg_metadata_dict['bbox'].append(rescale_bboxes(outputs['obj_boxes'][0][i].tolist(), im.size))
        kg_metadata_dict['extra_fields']['pred_labels'].append(probas_obj[i].argmax().item())
        kg_metadata_dict['extra_fields']['pred_scores'].append(probas_obj[i][probas_obj[i].argmax()].item())

    for i in range(0, len(probas)):
        kg_metadata_dict['extra_fields']['rel_pair_idxs'].append([i, i+len(probas)])
        kg_metadata_dict['extra_fields']['pred_rel_scores'].append(probas[i].tolist())
        kg_metadata_dict['extra_fields']['pred_rel_labels'].append(probas[i].argmax().item())

    img_filename = img_path.split("/")[-1].split(".")[0]
    output_filename = img_filename+"_output_sgg.json"

    output_path = os.path.join("./data/scene_graphs/", output_filename)

    with open(output_path, "w") as outfile: 
        json.dump(kg_metadata_dict, outfile)
    
    # sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)

    # print(outputs['obj_boxes'][0])
    # print("="*30)
    # print(keep)
    # print("="*30)
    # print(sub_bboxes_scaled)
    # # convert boxes from [0; 1] to image scales
    # sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'], im.size)
    # obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'], im.size)

    # topk = 10 # display up to 10 images
    # keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    # indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
    # keep_queries = keep_queries[indices]

    # # save the attention weights
    # conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []
    # hooks = [
    #     model.backbone[-2].register_forward_hook(
    #         lambda self, input, output: conv_features.append(output)
    #     ),
    #     model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
    #         lambda self, input, output: dec_attn_weights_sub.append(output[1])
    #     ),
    #     model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
    #         lambda self, input, output: dec_attn_weights_obj.append(output[1])
    #     )]

    # with torch.no_grad():
    #     # propagate through the model
    #     outputs = model(img)

    #     for hook in hooks:
    #         hook.remove()

    #     # don't need the list anymore
    #     conv_features = conv_features[0]
    #     dec_attn_weights_sub = dec_attn_weights_sub[0]
    #     dec_attn_weights_obj = dec_attn_weights_obj[0]

    #     # get the feature map shape
    #     h, w = conv_features['0'].tensors.shape[-2:]
    #     im_w, im_h = im.size

    #     fig, axs = plt.subplots(ncols=len(indices), nrows=3, figsize=(22, 7))
    #     for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
    #             zip(keep_queries, axs.T, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
    #         ax = ax_i[0]
    #         ax.imshow(dec_attn_weights_sub[0, idx].view(h, w))
    #         ax.axis('off')
    #         ax.set_title(f'query id: {idx.item()}')
    #         ax = ax_i[1]
    #         ax.imshow(dec_attn_weights_obj[0, idx].view(h, w))
    #         ax.axis('off')
    #         ax = ax_i[2]
    #         ax.imshow(im)
    #         ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
    #                                     fill=False, color='blue', linewidth=2.5))
    #         ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
    #                                     fill=False, color='orange', linewidth=2.5))

    #         ax.axis('off')
    #         ax.set_title(CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+' '+CLASSES[probas_obj[idx].argmax()], fontsize=10)

    #     fig.tight_layout()
    #     plt.show() # show the output

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    predict_sgg(args.img_path, args.resume)
