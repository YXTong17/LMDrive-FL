import math
import copy
import logging
from typing import Optional, List
from collections import OrderedDict
from functools import partial
from peft import LoraConfig, get_peft_model
import numpy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .registry import register_model
from .resnet import resnet26d, resnet50d, resnet18d, resnet26, resnet50, resnet101d, resnet34d
from .layers import StdConv2dSame, StdConv2d, to_2tuple
from .pointpillar import PointPillarNet, ConvBackbone
from lavis.models.blip2_models.blip2 import Blip2Base
from transformers import LlamaTokenizer
from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
_logger = logging.getLogger(__name__)

#迁移LayerNorm
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)





class Memfuser_Edge_Server(nn.Module):
    def __init__(
        self,
        num_features=None,
    ):
        super().__init__()
        

        from transformers import AutoTokenizer
        from transformers import AutoModelForCausalLM
        #将模型Drive Model的LLM Init部分迁移过来
        self.tokenizer = Blip2Base.init_tokenizer(truncation_side="left")
        max_txt_len=128
        self.num_features = num_features
        #这些布尔变量，暂时全是True
        self.has_qformer = True
        self.has_gru_decoder = False
        self.has_lora = True
        self.use_extra_prompt = False
        self.use_notice_prompt = False
        #需要模型路径的地址变量
        llm_model = "/home/tyx/yjl/LLamaTest/model_saved"


        self.split_section_num_for_visual_encoder = 1 #save gpu memory       
        self.ln_vision = LayerNorm(self.num_features)
        if 'opt' in llm_model:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side='left')
            self.llm_model = OPTForCausalLM.from_pretrained(llm_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        else:
            self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
            self.llm_model = LlamaForCausalLM.from_pretrained(llm_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)


        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        if self.has_gru_decoder:
            self.waypoints_fc = nn.Sequential(
                        nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size),
                        nn.ReLU(),
                        nn.Linear(self.llm_model.config.hidden_size, 64)
            )
            self.waypoints_predictor = nn.GRUCell(input_size=2, hidden_size=64)
            self.waypoints_output = nn.Linear(64, 2)
        else:
            self.waypoints_predictor = nn.Sequential(
                            nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size),
                            nn.ReLU(),
                            nn.Linear(self.llm_model.config.hidden_size, 10)
            )
        self.end_predictor = nn.Sequential(
            nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.llm_model.config.hidden_size, 2)
        )

        if self.has_qformer:
            print('Loading Q-Former')
            self.Qformer, self.query_tokens = Blip2Base.init_Qformer(
                4, self.num_features
            )
            self.Qformer.resize_token_embeddings(len(self.llm_tokenizer))
            self.Qformer.cls = None


        if self.has_lora:
            loraconfig = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj","v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llm_model = get_peft_model(self.llm_model, loraconfig)
            self.llm_model.print_trainable_parameters()
        else:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len

        self.waypoints_loss = torch.nn.L1Loss()
        self.end_loss = torch.nn.CrossEntropyLoss()






    def reset_parameters(self):
        nn.init.uniform_(self.global_embed)
        nn.init.uniform_(self.view_embed)
        nn.init.uniform_(self.query_embed)
        nn.init.uniform_(self.query_pos_embed)

    def forward_features(
        self,
        front_image,
        left_image,
        right_image,
        rear_image,
        front_center_image,
        lidar,
        num_points,
    ):
        features = []

        # Front view processing
        front_image_token, front_image_token_global = self.rgb_patch_embed(front_image)
        if self.use_view_embed:
            front_image_token = (
                front_image_token
                + self.view_embed[:, :, 0:1, :]
                + self.position_encoding(front_image_token)
            )
        else:
            front_image_token = front_image_token + self.position_encoding(
                front_image_token
            )
        front_image_token = front_image_token.flatten(2).permute(2, 0, 1)
        front_image_token_global = (
            front_image_token_global
            + self.view_embed[:, :, 0, :]
            + self.global_embed[:, :, 0:1]
        )
        front_image_token_global = front_image_token_global.permute(2, 0, 1)
        features.extend([front_image_token, front_image_token_global])

        if self.with_right_left_sensors:
            # Left view processing
            left_image_token, left_image_token_global = self.rgb_patch_embed(left_image)
            if self.use_view_embed:
                left_image_token = (
                    left_image_token
                    + self.view_embed[:, :, 1:2, :]
                    + self.position_encoding(left_image_token)
                )
            else:
                left_image_token = left_image_token + self.position_encoding(
                    left_image_token
                )
            left_image_token = left_image_token.flatten(2).permute(2, 0, 1)
            left_image_token_global = (
                left_image_token_global
                + self.view_embed[:, :, 1, :]
                + self.global_embed[:, :, 1:2]
            )
            left_image_token_global = left_image_token_global.permute(2, 0, 1)

            # Right view processing
            right_image_token, right_image_token_global = self.rgb_patch_embed(
                right_image
            )
            if self.use_view_embed:
                right_image_token = (
                    right_image_token
                    + self.view_embed[:, :, 2:3, :]
                    + self.position_encoding(right_image_token)
                )
            else:
                right_image_token = right_image_token + self.position_encoding(
                    right_image_token
                )
            right_image_token = right_image_token.flatten(2).permute(2, 0, 1)
            right_image_token_global = (
                right_image_token_global
                + self.view_embed[:, :, 2, :]
                + self.global_embed[:, :, 2:3]
            )
            right_image_token_global = right_image_token_global.permute(2, 0, 1)

            features.extend(
                [
                    left_image_token,
                    left_image_token_global,
                    right_image_token,
                    right_image_token_global,
                ]
            )

        if self.with_center_sensor:
            # Front center view processing
            (
                front_center_image_token,
                front_center_image_token_global,
            ) = self.rgb_patch_embed(front_center_image)
            if self.use_view_embed:
                front_center_image_token = (
                    front_center_image_token
                    + self.view_embed[:, :, 3:4, :]
                    + self.position_encoding(front_center_image_token)
                )
            else:
                front_center_image_token = (
                    front_center_image_token
                    + self.position_encoding(front_center_image_token)
                )

            front_center_image_token = front_center_image_token.flatten(2).permute(
                2, 0, 1
            )
            front_center_image_token_global = (
                front_center_image_token_global
                + self.view_embed[:, :, 3, :]
                + self.global_embed[:, :, 3:4]
            )
            front_center_image_token_global = front_center_image_token_global.permute(
                2, 0, 1
            )
            features.extend([front_center_image_token, front_center_image_token_global])


        if self.with_rear_sensor:
            # Rear view processing
            (
                rear_image_token,
                rear_image_token_global,
            ) = self.rgb_patch_embed(rear_image)
            if self.use_view_embed:
                rear_image_token = (
                    rear_image_token
                    + self.view_embed[:, :, 4:5, :]
                    + self.position_encoding(rear_image_token)
                )
            else:
                rear_image_token = (
                    rear_image_token
                    + self.position_encoding(rear_image_token)
                )

            rear_image_token = rear_image_token.flatten(2).permute(
                2, 0, 1
            )
            rear_image_token_global = (
                rear_image_token_global
                + self.view_embed[:, :, 4, :]
                + self.global_embed[:, :, 5:6]
            )
            rear_image_token_global = rear_image_token_global.permute(
                2, 0, 1
            )
            features.extend([rear_image_token, rear_image_token_global])

        lidar_token = self.lidar_backbone(lidar, num_points) # Batchsize * embed_dim * 50 * 50
        lidar_token = (
            lidar_token
            + self.position_encoding(lidar_token)
        )
        lidar_token = lidar_token.flatten(2).permute(2, 0, 1)

        features = torch.cat(features, 0)
        return features, lidar_token

    def forward(self, x):

        device = x['device']
        bs_llm = x['bs_llm']
        t  = x['t']
        # temp = dict()
        #随着客户端传过来的参数：device bs_llm t


        image_embeds = x['image_embeds']
        image_embeds = self.ln_vision(image_embeds)
        if self.has_qformer:
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            text_Qformer = self.llm_tokenizer(
                [i for i in x['text_input'] for _ in range(t)],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)

            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
            
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        image_embeds = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        # print("当前的bs大小：",bs_llm,"当前的t",t)
        image_embeds = image_embeds.view(bs_llm, t, *image_embeds.size()[1:])

        if self.use_extra_prompt:
            text_before_img = x['text_before_img']
            text_after_img = x['text_after_img']
            image_embeds, image_atts, end_flag_pos_list = self.prompt_wrap(image_embeds, text_before_img, text_after_img, x['valid_frames'])
        else:
            image_atts = None
            end_flag_pos_list = []
            n_length = image_embeds.size(2) # token number for each frame
            for i in range(bs_llm):
                end_flag_pos_list.append([n_length*(j+1)-1 for j in range(x['valid_frames'][i])])

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            x['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(device)
        with torch.no_grad():#冻结大模型，不知道好使不好使
            inputs_embeds = self.llm_model.get_input_embeddings()(text_input_tokens.input_ids)
            # inputs_embeds shape: (batch_size, sequence_length, hidden_size)

            if self.use_notice_prompt:
                llm_inputs, llm_attention_mask, input_part_targets_len, wp_target_index = self.concat_text_image_input_with_notice(inputs_embeds, text_input_tokens.attention_mask,
                                                                                                                    image_embeds, x['valid_frames'], end_flag_pos_list,
                                                                                                                    x['notice_frame_id'], x['notice_text'], image_atts)
            else:
                llm_inputs, llm_attention_mask, input_part_targets_len, wp_target_index = self.concat_text_image_input(inputs_embeds, text_input_tokens.attention_mask,
                                                                                                                    image_embeds, x['valid_frames'], end_flag_pos_list, image_atts)
            wp_target_index = torch.tensor(wp_target_index, device=device).long()
            from torch.cuda.amp import autocast
            with autocast():
                hidden_states = self.llm_model(
                        inputs_embeds=llm_inputs,
                        attention_mask=llm_attention_mask,
                        return_dict=False,
                    )
        # predicted_waypoints: bs, seq_len, 10
        if self.has_gru_decoder:
            # x['target_point'] = temp['target_point']#还原targets_point维度
            

            output_wp = []
            _, n_tokens, _ =hidden_states.size()
            x = torch.zeros(size=(bs_llm*n_tokens, 2), dtype=hidden_states.dtype).to(device)
            target_point = x['target_point'].view(bs_llm, -1, 2).to(device)

            target_point_list = []
            for i in range(bs_llm):
                target_point_list.append(target_point[i, :x['valid_frames'][i], :])
            target_point = torch.cat(target_point_list, 0)


            target_point_zeros = torch.zeros(size=(bs_llm, n_tokens, 2), dtype=hidden_states.dtype).to(device)
            target_point_zeros[wp_target_index[:,0], wp_target_index[:, 1]] = target_point.to(hidden_states.dtype)
            target_point_zeros = target_point_zeros.view(bs_llm*n_tokens, 2)
            target_point = target_point_zeros

            waypoints_feature = self.waypoints_fc(hidden_states.reshape(-1, self.llm_model.config.hidden_size))
            for _ in range(5):
                x_in = x# + target_point
                waypoints_feature = self.waypoints_predictor(x_in, waypoints_feature)
                dx = self.waypoints_output(waypoints_feature)
                x = dx + x
                output_wp.append(x)
            predicted_waypoints = torch.cat(output_wp, dim=1)
            predicted_waypoints = predicted_waypoints.view(bs_llm, n_tokens, 10)

        else:
            predicted_waypoints = self.waypoints_predictor(hidden_states)
            # predicted_waypoints: N * 10
        predicted_waypoints = predicted_waypoints[wp_target_index[:,0], wp_target_index[:, 1]]
        predicted_end_prob = self.end_predictor(hidden_states)
        predicted_end_prob = predicted_end_prob[wp_target_index[:,0], wp_target_index[:, 1]]


        gt_waypoints = self.build_gt_waypoints(x['local_future_waypoints'], x['valid_frames'])
        waypoints_loss = self.waypoints_loss(predicted_waypoints, gt_waypoints)






        return waypoints_loss
    def concat_text_image_input(self, input_embeds, input_atts, image_embeds, image_nums, end_flag_pos_list, image_atts=None):
        '''
        attention_mask:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        '''
        input_part_targets_len = []
        llm_inputs = []
        llm_attention_mask = []
        wp_target_index = []
        bs = image_embeds.size()[0]
        for i in range(bs):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            if image_atts is None:
                bs, t, n, dim = image_embeds.size()
                llm_inputs.append(
                    torch.cat([
                        input_embeds[i][:this_input_ones],
                        image_embeds[i].view(t*n, -1),
                        input_embeds[i][this_input_ones:]
                    ])
                )
            else:
                llm_inputs.append(
                    torch.cat([
                        input_embeds[i][:this_input_ones],
                        image_embeds[i],
                        input_embeds[i][this_input_ones:]
                    ])
                )
            if image_atts is None:
                bs, t, n, dim = image_embeds.size()
                llm_attention_mask.append(
                    torch.cat([
                        input_atts[i][:this_input_ones],
                        torch.ones((image_nums[i]*n), device=image_embeds.device, dtype=torch.long),
                        torch.zeros(((t-image_nums[i])*n), device=image_embeds.device, dtype=torch.long),
                        input_atts[i][this_input_ones:]
                    ])
                )
            else:
                llm_attention_mask.append(
                    torch.cat([
                        input_atts[i][:this_input_ones],
                        image_atts[i],
                        input_atts[i][this_input_ones:]
                    ])
                )
            sub_target_index = []
            for j in end_flag_pos_list[i]:
                sub_target_index.append([i, j + this_input_ones])
            wp_target_index.extend(sub_target_index)
        llm_inputs = torch.stack(llm_inputs, 0)
        llm_attention_mask = torch.stack(llm_attention_mask, 0)
        return llm_inputs, llm_attention_mask, input_part_targets_len, wp_target_index

    def concat_text_image_input_with_notice(self, input_embeds, input_atts, image_embeds, image_nums,
                                            end_flag_pos_list, notice_frame_id, notice_text, image_atts=None):
        '''
        the function is made for processing data with [inserted] notice text
        notice_frame_id: how many image frames before the notice
        attention_mask:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        '''
        input_part_targets_len = []
        llm_inputs = []
        llm_attention_mask = []
        wp_target_index = []
        bs = image_embeds.size()[0]

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            notice_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image_embeds.device)
        input_notice_atts = text_input_tokens.attention_mask
        notice_embeds = self.llm_model.get_input_embeddings()(text_input_tokens.input_ids)

        for i in range(bs):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)

            this_notice_input_ones = input_notice_atts[i].sum()
            if image_atts is None:
                bs, t, n, dim = image_embeds.size()
                if notice_frame_id[i] <= 0: # which means the scenario do not include any notice
                    llm_inputs.append(
                        torch.cat([
                            input_embeds[i][:this_input_ones],
                            image_embeds[i].view(t*n, -1),
                            input_embeds[i][this_input_ones:],
                            notice_embeds[i][:],
                        ])
                    )
                else:
                    llm_inputs.append(
                        torch.cat([
                            input_embeds[i][:this_input_ones],
                            image_embeds[i, :notice_frame_id[i]].view(notice_frame_id[i]*n, -1),
                            notice_embeds[i][:this_notice_input_ones],
                            image_embeds[i, notice_frame_id[i]:].view((t-notice_frame_id[i])*n, -1),
                            input_embeds[i][this_input_ones:],
                            notice_embeds[i][this_notice_input_ones:],
                        ])
                    )
            else:
                pass #TODO
            if image_atts is None:
                bs, t, n, dim = image_embeds.size()
                if notice_frame_id[i] < 0: # which means the scenario do not include any notice
                    llm_attention_mask.append(
                        torch.cat([
                            input_atts[i][:this_input_ones],
                            torch.ones((image_nums[i]*n), device=image_embeds.device, dtype=torch.long),
                            torch.zeros(((t-image_nums[i])*n), device=image_embeds.device, dtype=torch.long),
                            torch.zeros((input_notice_atts.size(1)), device=image_embeds.device, dtype=torch.long),
                            input_atts[i][this_input_ones:]
                        ])
                    )
                else:
                    llm_attention_mask.append(
                        torch.cat([
                            input_atts[i][:this_input_ones],
                            torch.ones((image_nums[i]*n), device=image_embeds.device, dtype=torch.long),
                            input_notice_atts[i][:this_notice_input_ones],
                            torch.zeros(((t-image_nums[i])*n), device=image_embeds.device, dtype=torch.long),
                            input_atts[i][this_input_ones:],
                            input_notice_atts[i][this_notice_input_ones:],
                        ])
                    )
            else:
                pass
            sub_target_index = []
            for j in range(len(end_flag_pos_list[i])):
                if j < notice_frame_id[i] or notice_frame_id[i] < 0: # when notice is '', the input_ones is 1, not ZERO
                    sub_target_index.append([i, end_flag_pos_list[i][j] + this_input_ones])
                else:
                    sub_target_index.append([i, end_flag_pos_list[i][j] + this_input_ones + this_notice_input_ones])
            wp_target_index.extend(sub_target_index)
        llm_inputs = torch.stack(llm_inputs, 0)
        llm_attention_mask = torch.stack(llm_attention_mask, 0)
        return llm_inputs, llm_attention_mask, input_part_targets_len, wp_target_index

    def build_gt_waypoints(self, waypoints, valid_frames):
        gt_waypoints = []
        for i in range(waypoints.size(0)):
            gt_waypoints.append(waypoints[i, :valid_frames[i]])
        gt_waypoints = torch.cat(gt_waypoints, dim=0)
        return gt_waypoints      

@register_model
def memfuser_baseline_e1d3_edge_server(num_features=None,**kwargs):
    model = Memfuser_Edge_Server(
        num_features=num_features
    )
    return model

