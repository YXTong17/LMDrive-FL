



#!/usr/bin/env python3
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from render import render, render_waypoints

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from tensorboardX import SummaryWriter
import time


import socket
import json
import pickle
def Client_to_Server():
    host,port = '0.0.0.0',20840
    time.sleep(3)
    Receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    Receiver_socket.bind((host, port))
    Receiver_socket.listen(1)
    print(f"Server listening on {host}:{port}")


    Sent_socket, addr = Receiver_socket.accept()
    print(f"Connection from {addr} has been established.")
    # 接收数据长度
    data_length_bytes = Sent_socket.recv(1024)
    data_length = int(data_length_bytes)
    print("接收到当前的数据长度",data_length)
    
    # 初始化接收的数据缓冲区
    received_data = b''

    # 循环接收数据直到达到指定长度
    while len(received_data) < data_length:
        chunk = Sent_socket.recv(1024)
        if not chunk:
            break
        received_data += chunk

    # 使用pickle反序列化数据
    received_dict = pickle.loads(received_data)
    Sent_socket.close()
    Receiver_socket.close()
    return received_dict

    

def Server_to_Client(ServerOutput):
    Sent_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    Sent_socket.connect(('127.0.0.1', 20841))
    # 使用pickle序列化字典
    serialized_data = pickle.dumps(ServerOutput)

    # 发送数据长度
    Sent_socket.sendall(str(len(serialized_data)).encode('utf-8'))

    # 发送数据
    Sent_socket.sendall(serialized_data)

    # 接收服务端的响应
    response = Sent_socket.recv(1024)

    Sent_socket.close()


if __name__ == "__main__":
    # from timm.models import *
    # model = create_model(model_name="memfuser_baseline_e1d3_edge_server",pretrained=False)

    while True:
        """ 服务端开始监听函数，直到客户端向服务端发送数据"""
        ServerModel_Input = Client_to_Server() #从客户端的Client_ContinueToForword_Server获得ServerModel_Input
        from timm.models.memfuser_egde_server import memfuser_baseline_e1d3_edge_server

        model  = memfuser_baseline_e1d3_edge_server(num_features=ServerModel_Input['num_features']).cuda()
        ServerModel_Input = ServerModel_Input
        ServerOutput = model(ServerModel_Input)
        Server_to_Client(ServerOutput) #服务端前向传播得到的waypoint_loss传回客户端的Client_ContinueToForword_Server






    
