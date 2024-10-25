# import os
# import random
# import copy
# import re
# import logging
# from pathlib import Path

# import json
# import numpy as np
# import torch
# import torch.utils.data as data
# from PIL import Image
# from torchvision import transforms
# from torch.utils.data.dataloader import default_collate

# from .base_io_dataset import BaseIODataset
# from .transforms_carla_factory import create_carla_rgb_transform
# from timm.data.heatmap_utils import generate_heatmap, generate_future_waypoints, get_yaw_angle
# from timm.data.det_utils import generate_det_data
# from skimage.measure import block_reduce
# curr_dir = Path(__file__).parent
# instruction_json = os.path.join(curr_dir, "../../../../", "leaderboard/leaderboard/envs", 'instruction_dict.json')
# INSTRUCTION_DICT = json.load(open(instruction_json))


# _logger = logging.getLogger(__name__)

# def get_yaw_angle(forward_vector):
#     forward_vector = forward_vector / np.linalg.norm(forward_vector)
#     yaw = math.acos(forward_vector[0])
#     if forward_vector[1] < 0:
#         yaw = 2 * np.pi - yaw
#     return yaw


# def rotate_lidar(lidar, angle):
#     radian = np.deg2rad(angle)
#     return lidar @ [
#         [ np.cos(radian), np.sin(radian), 0, 0],
#         [-np.sin(radian), np.cos(radian), 0, 0],
#         [0,0,1,0],
#         [0,0,0,1]
#     ]

# def lidar_to_raw_features(lidar):
#     def preprocess(lidar_xyzr, lidar_painted=None):

#         idx = (lidar_xyzr[:,0] > -1.2)&(lidar_xyzr[:,0] < 1.2)&(lidar_xyzr[:,1]>-1.2)&(lidar_xyzr[:,1]<1.2)

#         idx = np.argwhere(idx)

#         if lidar_painted is None:
#             return np.delete(lidar_xyzr, idx, axis=0)
#         else:
#             return np.delete(lidar_xyzr, idx, axis=0), np.delete(lidar_painted, idx, axis=0)

#     lidar_xyzr = preprocess(lidar)

#     idxs = np.arange(len(lidar_xyzr))
#     np.random.shuffle(idxs)
#     lidar_xyzr = lidar_xyzr[idxs]

#     lidar = np.zeros((40000, 4), dtype=np.float32)
#     num_points = min(40000, len(lidar_xyzr))
#     lidar[:num_points,:4] = lidar_xyzr[:num_points]
#     lidar[np.isinf(lidar)] = 0
#     lidar[np.isnan(lidar)] = 0
#     lidar = rotate_lidar(lidar, -90).astype(np.float32)
#     return lidar, num_points

# def check_data(data, info):
#     for key in data:
#         if isinstance(data[key], np.ndarray):
#             if np.isnan(data[key]).any():
#                 print(key)
#                 print(info)
#                 data[key][np.isnan(data[key])] = 0
#             if np.isinf(data[key]).any():
#                 print(key)
#                 print(info)
#                 data[key][np.isinf(data[key])] = 0
#         elif isinstance(data[key], torch.Tensor):
#             if torch.isnan(data[key]).any():
#                 print(key)
#                 print(info)
#                 data[key][torch.isnan(data[key])] = 0
#             if torch.isinf(data[key]).any():
#                 print(key)
#                 print(info)
#                 data[key][torch.isinf(data[key])] = 0
#     return data


# class CarlaFLDataset(BaseIODataset):
#     def _get_frames_paths(self, root, weathers, towns):
#         route_frames = []
#         route_dir_nums = 0
#         dataset_indexs = self._load_text(os.path.join(root, 'dataset_index_test.txt')).split('\n')
#         pattern = re.compile('town(\d\d).*w(\d+)')
#         for line in dataset_indexs:
#             if len(line.split()) != 2:
#                 continue
#             path, frames = line.split()
#             path = os.path.join(root, 'data', path)
#             frames = int(frames)
#             res = pattern.findall(path)
#             if len(res) != 1:
#                 continue
#             town = int(res[0][0])
#             weather = int(res[0][1])
#             if weather not in weathers or town not in towns:
#                 continue
#             route_dir_nums += 1
#             for i in range(0, frames):
#                 route_frames.append((path, i, route_dir_nums))
#         _logger.info("Sub route dir nums: %d" % len(route_frames))
#         return route_frames
#     def __init__(
#         self,
#         dataset_root,
#         towns=None,
#         weathers=None,
#         scale=None,
#         is_training=False,
#         input_rgb_size=224,
#         input_multi_view_size=128,
#         input_lidar_size=224,
#         token_max_length=32,
#         sample_interval=2,
#         enable_start_frame_augment=False,
#         enable_notice=False,
#         **kwargs,
#     ):
#         super().__init__()
#         ####添加来自CarlaMVDetDataset代码：
#         self.head="det"
#         self.input_lidar_size = input_lidar_size
#         self.input_rgb_size = input_rgb_size
#         self.rgb_transform = None
#         self.seg_transform = None
#         self.depth_transform = None
#         self.lidar_transform = None
#         self.multi_view_transform = None

#         self.with_waypoints = None

#         self.with_seg = None
#         self.with_depth = None
#         self.with_lidar = None
#         self.multi_view = None

#         self.route_frames = self._get_frames_paths(dataset_root, weathers, towns)

#         #####
#         self.token_max_length = token_max_length
#         self.rgb_transform = create_carla_rgb_transform(
#             input_rgb_size,
#             is_training=is_training,
#             scale=scale,
#         )
#         self.rgb_center_transform = create_carla_rgb_transform(
#             128,
#             scale=None,
#             is_training=is_training,
#             need_scale=False,
#         )
#         self.multi_view_transform = create_carla_rgb_transform(
#             input_multi_view_size,
#             scale=scale,
#             is_training=is_training,
#         )

#         self.scenario_infos = self._get_scenario_paths(dataset_root, weathers, towns)
#         _logger.info("Scenario nums: %d" % len(self.scenario_infos))
#         self.instruction_dict = INSTRUCTION_DICT
#         self.sample_interval = sample_interval
#         self.enable_start_frame_augment = enable_start_frame_augment
#         self.enable_notice = enable_notice
#         if self.enable_notice:
#             raw_notice_data = self._load_json(os.path.join(dataset_root, 'notice_instruction_list.json'))
#             self.notice_data = {}
#             for key in raw_notice_data:
#                 self.notice_data[os.path.join(dataset_root, key)] = raw_notice_data[key]

#     def collater(self, samples):
#         return default_collate(samples)

#     def _get_scenario_paths(self, dataset_root, weathers, towns):
#         scenario_infos = []
#         dataset_indexs = self._load_text(os.path.join(dataset_root, 'navigation_instruction_list.txt')).split('\n')
#         for line in dataset_indexs:
#             if len(line) < 10: continue
#             info = json.loads(line.strip())
#             # result {dict}: route_path, town_id, weather_id, start_frame, end_frame, instruction, instruction_id, instruction_args, route_frames
#             if towns is not None:
#                 if info['town_id'] not in towns:
#                     continue
#             if weathers is not None:
#                 if info['weather_id'] not in weathers:
#                     continue
#             info['route_path'] = os.path.join(dataset_root, info['route_path'])
#             scenario_infos.append(info)
#         return scenario_infos

#     def __len__(self):
#         return len(self.scenario_infos)

#     def pad_and_stack(self, data):
#         if isinstance(data[0], np.ndarray):
#             for _ in range(self.token_max_length - len(data)):
#                 data.append(np.zeros_like(data[0]))
#             data = np.stack(data, 0)
#         elif torch.is_tensor(data[0]):
#             for _ in range(self.token_max_length - len(data)):
#                 data.append(torch.zeros_like(data[0]))
#             data = torch.stack(data, 0)
#         else:
#             for _ in range(self.token_max_length - len(data)):
#                 data.append(0)
#             data = np.array(data).reshape(-1)
#         return data

#     def __getitem__(self, idx):
#         info = self.scenario_infos[idx]
#         route_path = info['route_path']
#         route_frames = int(info['route_frames'])
#         town_id = info['town_id']
#         weather_id = info['weather_id']


#         if 'Turn' in info['instruction']:
#             info['end_frame'] = min(route_frames - 1, info['end_frame'] + 12)

#         sample_interval = self.sample_interval

#         start_frame_id = info['start_frame'] + random.randint(0, sample_interval)
#         if self.enable_start_frame_augment and len(info['instruction_args']) == 0: # if instruction_args has no values, it means the instruction doesn't include distance
#             if 'Other' not in info['instruction'] or info['instruction'] != 'Follow-01':
#                 augment_range = min(16, max(0, self.token_max_length * self.sample_interval - (info['end_frame'] - info['start_frame'])))
#                 start_frame_id = max(0, start_frame_id - random.randint(0, augment_range))
#         end_frame_id = min(info['end_frame'] + 1, start_frame_id + self.token_max_length * sample_interval - random.randint(0, sample_interval-1))

#         # we construct notice data after obtaining the final start/end frame id
#         if self.enable_notice:
#             notice_frame_id = []
#             notice_text = []
#             if route_path in self.notice_data:
#                 notice_list = self.notice_data[route_path]
#                 notice_list = [x for x in notice_list if x['frame_id'] > start_frame_id and x['frame_id'] < end_frame_id - 16]
#                 if len(notice_list) < 1 or random.random() < 0.75:
#                     notice_frame_id = -1
#                     notice_text = ''
#                 else:
#                     notice = random.choice(notice_list)
#                     # we convert the abslote poisition to the relative position
#                     notice_frame_id = (notice['frame_id'] - start_frame_id) // sample_interval + 1
#                     notice_text = np.random.choice(self.instruction_dict[str(notice['instruction_id'])])
#             else:
#                 notice_frame_id = -1
#                 notice_text = ''

#         measurements = self._load_json(os.path.join(route_path, "measurements_all.json"))
#         ego_theta = measurements[start_frame_id]['theta']
#         processed_data = {}

#         if np.isnan(ego_theta):
#             ego_theta = 0
#         R = np.array(
#             [[np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
#             [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)]])
#         origin_x = measurements[start_frame_id]['gps_x']
#         origin_y = measurements[start_frame_id]['gps_y']


#         ego_throttles = []
#         ego_steers = []
#         ego_brakes = []
#         ego_velocitys = []
#         ego_xs = []
#         ego_ys = []
#         local_positions = []
#         local_future_waypoints = []
#         text_before_img = []
#         text_after_img = []
#         target_points = []

#         for frame_id in range(start_frame_id, end_frame_id, sample_interval):
#             ego_x = measurements[frame_id]['gps_x']
#             ego_y = measurements[frame_id]['gps_y']
#             velocity = measurements[frame_id]['speed']
#             local_position = np.array([ego_x - origin_x, ego_y - origin_y])
#             local_position = R.T.dot(local_position)
#             text_before_img.append('<frame %.1f,%.1f>' % (local_position[0], local_position[1]))
#             # text_before_img.append('<frame=%d;x=%.1f;y=%.1f;speed=%.1f>' % (frame_id-start_frame_id, local_position[0], local_position[1], velocity))
#             text_after_img.append('</frame>')

#             ego_xs.append(ego_x)
#             ego_ys.append(ego_y)
#             ego_throttles.append(measurements[frame_id]['throttle'])
#             ego_steers.append(measurements[frame_id]['steer'])
#             ego_brakes.append(int(measurements[frame_id]['brake']))
#             ego_velocitys.append(velocity)
#             local_positions.append(local_position.reshape(-1))
#             local_ego_theta = measurements[frame_id]['theta']
#             if np.isnan(local_ego_theta):
#                 local_ego_theta = 0
#             local_R = np.array(
#                 [[np.cos(np.pi / 2 + local_ego_theta), -np.sin(np.pi / 2 + local_ego_theta)],
#                 [np.sin(np.pi / 2 + local_ego_theta), np.cos(np.pi / 2 + local_ego_theta)]])

#             x_command = measurements[frame_id]["x_command"]
#             y_command = measurements[frame_id]["y_command"]
#             local_command_point = np.array([x_command - ego_x, y_command - ego_y])
#             local_command_point = local_R.T.dot(local_command_point)
#             if any(np.isnan(local_command_point)):
#                 local_command_point[np.isnan(local_command_point)] = np.mean(
#                     local_command_point
#                 )
#             local_command_point = local_command_point.reshape(-1)
#             target_points.append(local_command_point)

#             local_future_waypoints_temp = []
#             for future_frame_delta in range(1, 6):
#                 future_frame_id = min(frame_id + future_frame_delta * 5, route_frames-1)
#                 future_ego_x = measurements[future_frame_id]['gps_x']
#                 future_ego_y = measurements[future_frame_id]['gps_y']
#                 future_waypoint = np.array([future_ego_x - ego_x, future_ego_y - ego_y])
#                 future_waypoint = local_R.T.dot(future_waypoint)
#                 local_future_waypoints_temp.append(future_waypoint.reshape(1, 2))
#             local_future_waypoints.append(np.concatenate(local_future_waypoints_temp, axis=0).reshape(-1))

#         valid_frames = len(ego_xs)
#         ego_throttles = self.pad_and_stack(ego_throttles)
#         ego_steers = self.pad_and_stack(ego_steers)
#         ego_brakes = self.pad_and_stack(ego_brakes)
#         ego_velocitys = self.pad_and_stack(ego_velocitys)
#         ego_xs = self.pad_and_stack(ego_xs)
#         ego_ys = self.pad_and_stack(ego_ys)
#         target_points = self.pad_and_stack(target_points)
#         local_positions = self.pad_and_stack(local_positions)
#         local_future_waypoints = self.pad_and_stack(local_future_waypoints)

#         lidar_data = []
#         lidar_num_points = []
#         rgb_front = []
#         rgb_center = []
#         rgb_left = []
#         rgb_right = []
#         rgb_rear = []
#         measurement_data = []
#         heatmap_mask  = []
#         command = []
#         targets = []
#         targets_flag = False
#         for frame_id in range(start_frame_id, end_frame_id, sample_interval):
#             sensor_data,target = self._extract_data_item(route_path, frame_id)
#             lidar_data.append(sensor_data['lidar'])
#             lidar_num_points.append(sensor_data['num_points'])
#             rgb_front.append(sensor_data['rgb'])
#             rgb_center.append(sensor_data['rgb_center'])
#             rgb_left.append(sensor_data['rgb_left'])
#             rgb_right.append(sensor_data['rgb_right'])
#             rgb_rear.append(sensor_data['rgb_rear'])
#             measurement_data.append(sensor_data['measurements'])
#             heatmap_mask.append(sensor_data['heatmap_mask'])
#             command.append(sensor_data['command'])
#             if targets_flag == False:
#                 targets = target
#                 for i in range(len(targets)):#限制曾在vision encoder中出现的targets改shape
#                     # print("当前i是：",i)
#                     if isinstance(targets[i], int):
#                         # print("当前是第",i,"从int变成tensor")
#                         targets[i] = [targets[i]]
#                         targets[i] =  torch.tensor(targets[i])
#                     elif isinstance(targets[i], np.ndarray):
#                         targets[i] =  torch.from_numpy(targets[i])
#                     if  not isinstance(targets[i], torch.Tensor):
#                         targets[i] = torch.tensor(targets[i])
                        
#                     if i in [1,4]:
#                         targets[i] = torch.reshape( targets[i], (1,  targets[i].size(0),  targets[i].size(1)))
#                     elif i in [2,3,6]:
#                         targets[i] = torch.reshape( targets[i], (1,  targets[i].size(0)))

#                 targets_flag = True  
#             else:
#                 # print(len(targets))
#                 for i in range(len(targets)):
#                     # print(i)
#                     # print("当前是第",i,'个的targets:',targets[i])
#                     # print("当前是第",i,'个的target:',target[i])
#                     if isinstance(target[i], int):
#                         target[i] = [target[i]]
#                         target[i] =  torch.tensor(target[i])
#                     elif isinstance(target[i], np.ndarray):
#                         target[i] =  torch.from_numpy(target[i])


#                     if i in [1,4]:
#                         target[i] = torch.reshape( target[i], (1,  target[i].size(0),  target[i].size(1)))
#                     elif i in [2,3,6]:
#                         target[i] = torch.reshape( target[i], (1,  target[i].size(0)))
#                     if targets[i].dim() != 0 and target[i].dim() != 0:
#                         targets[i] = torch.cat((targets[i], target[i]), dim=0)

#                     # print("当前是第",i,'个的targets:',targets[i].size())
#                     # print("当前是第",i,'个的target:',target[i].size())









#         processed_data['lidar'] = self.pad_and_stack(lidar_data)
#         processed_data['num_points'] = self.pad_and_stack(lidar_num_points)
#         processed_data['rgb_front'] = self.pad_and_stack(rgb_front)
#         processed_data['rgb_left'] = self.pad_and_stack(rgb_left)
#         processed_data['rgb_right'] = self.pad_and_stack(rgb_right)
#         processed_data['rgb_rear'] = self.pad_and_stack(rgb_rear)
#         processed_data['rgb_center'] = self.pad_and_stack(rgb_center)
#         processed_data['measurements'] = self.pad_and_stack(measurement_data)
#         processed_data['heatmap_mask'] = self.pad_and_stack(heatmap_mask)
#         processed_data['command'] = self.pad_and_stack(command)

#         for t in range(len(target)):
#             if target[t].dim() != 0:
#                 target[t] = self.pad_and_stack(list(target[t]))
#         processed_data['target'] = target
#         # print("*****************",processed_data['target'][1].shape)


#         instruction_text = np.random.choice(self.instruction_dict[str(info['instruction_id'])])
#         try:
#             if '[x]' in instruction_text:
#                 instruction_text.replace('[x]', str(info['instruction_args'][0]))
#             if 'left/right' in instruction_text:
#                 instruction_text.replace('left/right', str(info['instruction_args'][1]))
#             if '[y]' in instruction_text:
#                 instruction_text.replace('[y]', str(info['instruction_args'][2]))
#         except Exception as e:
#             _logger.error(e)
#             _logger.info(info)
#             _logger.info(instruction_text)

#         processed_data['target_point'] = torch.from_numpy(target_points).float()
#         processed_data['valid_frames'] = valid_frames
#         processed_data['text_input'] = instruction_text
#         processed_data['text_before_img'] = '|'.join(text_before_img)
#         processed_data['text_after_img'] = '|'.join(text_after_img)
#         processed_data['ego_throttles'] = ego_throttles
#         processed_data['ego_steers'] = ego_steers
#         processed_data['ego_brakes'] = ego_brakes
#         processed_data['velocity'] = torch.from_numpy(np.array(ego_velocitys)).float()
#         processed_data['local_positions'] = local_positions
#         processed_data['local_future_waypoints'] = local_future_waypoints
#         if self.enable_notice:
#             processed_data['notice_frame_id'] = notice_frame_id
#             processed_data['notice_text'] = notice_text

#         return processed_data


#     def _extract_data_item(self, route_path, frame_id):
#         data = {}
#         # You can use tools/data/batch_merge_data.py to generate FULL image (including front, left, right) for reducing io cost
#         rgb_full_image = self._load_image(
#             os.path.join(route_path, "rgb_full", "%04d.jpg" % frame_id)
#         )
#         rgb_image = rgb_full_image.crop((0, 0, 800, 600))
#         rgb_left_image = rgb_full_image.crop((0, 600, 800, 1200))
#         rgb_right_image = rgb_full_image.crop((0, 1200, 800, 1800))
#         rgb_rear_image = rgb_full_image.crop((0, 1800, 800, 2400))

#         '''
#         rgb_image = self._load_image(
#             os.path.join(route_path, "rgb_front", "%04d.jpg" % frame_id)
#         )
#         rgb_left_image = self._load_image(
#             os.path.join(route_path, "rgb_left", "%04d.jpg" % frame_id)
#         )
#         rgb_right_image = self._load_image(
#             os.path.join(route_path, "rgb_right", "%04d.jpg" % frame_id)
#         )
#         '''
#         #####代码添加####
#         measurements_all = self._load_json(
#                 os.path.join(route_path, "measurements_all.json")
#             )
#         measurements = self._load_json(
#                 os.path.join(route_path, "measurements_full", "%04d.json" % frame_id)
#             )
#         actors_data = measurements["actors_data"]
#         stop_sign = int(measurements["stop_sign"])
#         if measurements["is_junction"] is True:
#             is_junction = 1
#         else:
#             is_junction = 0

#         if len(measurements['is_red_light_present']) > 0:
#             traffic_light_state = 0
#         else:
#             traffic_light_state = 1
#         #####
#         lidar_unprocessed_front = self._load_npy(
#             os.path.join(route_path, "lidar", "%04d.npy" % frame_id)
#         )[..., :4]
#         lidar_unprocessed_back = self._load_npy(
#             os.path.join(route_path, "lidar_odd", "%04d.npy" % max(frame_id - 1, 0))
#         )[..., :4]
#         lidar_unprocessed = np.concatenate([lidar_unprocessed_front, lidar_unprocessed_back])
#         lidar_processed, num_points= lidar_to_raw_features(lidar_unprocessed)
#         data['lidar'] = lidar_processed
#         data['num_points'] = num_points
#         ####添加代码####
#         cmd_one_hot = [0, 0, 0, 0, 0, 0]
#         cmd = measurements["command"] - 1
#         if cmd < 0:
#             cmd = 3
#         cmd_one_hot[cmd] = 1
#         cmd_one_hot.append(measurements["speed"])
#         mes = np.array(cmd_one_hot)
#         mes = torch.from_numpy(mes).float()

#         data["measurements"] = mes
#         data['velocity'] = torch.from_numpy(np.array([measurements['speed']])).float()
#         data['command'] = torch.from_numpy(np.array(cmd))

#         if np.isnan(measurements["theta"]):
#             measurements["theta"] = 0
#         ego_theta = measurements["theta"]
#         x_command = measurements["x_command"]
#         y_command = measurements["y_command"]
#         ego_x = measurements["gps_x"]
#         ego_y = measurements["gps_y"]
#         R = np.array(
#             [
#                 [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
#                 [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)],
#             ]
#         )
#         local_command_point = np.array([x_command - ego_x, y_command - ego_y])
#         local_command_point = R.T.dot(local_command_point)
#         if any(np.isnan(local_command_point)):
#             local_command_point[np.isnan(local_command_point)] = np.mean(
#                 local_command_point
#             )
#         local_command_point = torch.from_numpy(local_command_point).float()
#         data["target_point"] = local_command_point

#         command_waypoints = []
#         for i in range(5):
#             fid = min(len(measurements_all)-1, frame_id+5*(i+1))
#             waypoint = [measurements_all[fid]['gps_x'], measurements_all[fid]['gps_y']]
#             new_loc = R.T.dot(np.array([waypoint[0] - ego_x, waypoint[1] - ego_y]))
#             command_waypoints.append(new_loc.reshape(1, 2))
#         command_waypoints = np.concatenate(command_waypoints)
#         command_waypoints = torch.from_numpy(command_waypoints).float()





#         ####
#         if self.rgb_transform is not None:
#             rgb_main_image = self.rgb_transform(rgb_image)
#         data["rgb"] = rgb_main_image

#         if self.rgb_center_transform is not None:
#             rgb_center_image = self.rgb_center_transform(rgb_image)
#         data["rgb_center"] = rgb_center_image

#         if self.multi_view_transform is not None:
#             rgb_left_image = self.multi_view_transform(rgb_left_image)
#             rgb_right_image = self.multi_view_transform(rgb_right_image)
#             rgb_rear_image = self.multi_view_transform(rgb_rear_image)
#         data["rgb_left"] = rgb_left_image
#         data["rgb_right"] = rgb_right_image
#         data["rgb_rear"] = rgb_rear_image

#         ####代码添加####
#         heatmap = generate_heatmap(
#             copy.deepcopy(measurements), copy.deepcopy(actors_data)
#         )
#         det_data = (
#             generate_det_data(
#                 heatmap, copy.deepcopy(measurements), copy.deepcopy(actors_data)).reshape(2500, -1).astype(np.float32)
#             )

#         heatmap_mask = np.zeros((50, 50), dtype=bool).reshape(-1)
#         reduced_heatmap = block_reduce(heatmap[:250, 25:275], block_size=(5, 5), func=np.mean)
#         flattened_reduced_heatmap = reduced_heatmap.reshape(-1)
#         heatmap_mask[np.argsort(flattened_reduced_heatmap)[-10:]] = True
#         heatmap_mask = heatmap_mask.reshape(50, 50)
#         data['heatmap_mask'] = transforms.ToTensor()(heatmap_mask)

#         img_traffic = heatmap[:250, 25:275, None]
#         img_traffic = transforms.ToTensor()(img_traffic)

#         img_traj = generate_future_waypoints(measurements)
#         img_traj = img_traj[:250, 25:275, None]
#         img_traj = transforms.ToTensor()(img_traj)

#         reduced_heatmap = reduced_heatmap[:, :, None]
#         reduced_heatmap = transforms.ToTensor()(reduced_heatmap) / 255.
#         ####

#         data = check_data(data, info=route_path+str(frame_id))
#         return data, [
#             img_traffic,
#             command_waypoints,
#             is_junction,
#             traffic_light_state,
#             det_data,
#             img_traj,
#             stop_sign,
#             reduced_heatmap,
#         ]
