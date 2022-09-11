import socket
import sys
import time
import json

send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
send_socket.settimeout(1)
while True:
    time.sleep(0.1)
    try:
        send_socket.connect(("127.0.0.1", 30000))
    except socket.error:
        continue
    msg = {}
    msg["type"] = "initialize"

    # correct
    # adam_epsilon=1e-08, arrival_interval=3, async_buffer=10, async_mode=False, backbone='./resnet50.pth', backend='gloo', batch_size=20, bidirectional=True, blacklist_max_len=0.3, blacklist_rounds=-1, block_size=64, cfg_file='./utils/rcnn/cfgs/res101.yml', checkin_period=50, clf_block_size=32, clip_bound=0.9, clip_threshold=3.0, clock_factor=1.1624548736462095, conf_path='~/dataset/', connection_timeout=60, cuda_device=None, cut_off_util=0.05, data_cache='', data_dir='/users/yilegu/fedscale_k8s/FedScale/benchmark/dataset/data/femnist', data_map_file='/users/yilegu/fedscale_k8s/FedScale/benchmark/dataset/data/femnist/client_data_mapping/train.csv', data_set='femnist', decay_factor=0.98, decay_round=10, device_avail_file='/users/yilegu/fedscale_k8s/FedScale/benchmark/dataset/data/device_info/client_behave_trace', device_conf_file='/users/yilegu/fedscale_k8s/FedScale/benchmark/dataset/data/device_info/client_device_capacity', dump_epoch=10000000000.0, embedding_file='glove.840B.300d.txt', engine='pytorch', epsilon=0.9, eval_interval=30, executor_configs='localhost:[1]', experiment_mode='simulation', exploration_alpha=0.3, exploration_decay=0.98, exploration_factor=0.9, exploration_min=0.3, filter_less=21, filter_more=1000000000000000.0, finetune=False, gamma=0.9, gradient_policy='yogi', hidden_layers=7, hidden_size=256, input_dim=0, job_name='femnist', labels_path='labels.json', learning_rate=0.05, line_by_line=False, local_steps=20, log_path='/users/yilegu/fedscale_k8s/FedScale/benchmark', loss_decay=0.2, malicious_factor=4, memory_capacity=2000, min_learning_rate=5e-05, mlm=False, mlm_probability=0.15, model='shufflenet_v2_x2_0', model_size=65536, model_zoo='torchcv', n_actions=2, n_states=4, noise_dir=None, noise_factor=0.1, noise_max=0.5, noise_min=0.0, noise_prob=0.4, num_class=62, num_classes=35, num_executors=1, num_loaders=2, num_participants=20, output_dim=0, overcommitment=1.3, overwrite_cache=False, pacer_delta=5, pacer_step=20, proxy_mu=0.1, ps_ip='localhost', ps_port='29501', rnn_type='lstm', round_penalty=2.0, round_threshold=30, rounds=10, sample_mode='random', sample_rate=16000, sample_seed=233, sample_window=5.0, spec_augment=False, speed_volume_perturb=False, target_delta=0.0001, target_replace_iter=15, task='cv', test_bsz=20, test_manifest='data/test_manifest.csv', test_output_dir='./logs/server', test_ratio=1.0, test_size_file='', this_rank=0, time_stamp='0910_005427', train_manifest='data/train_manifest.csv', train_size_file='', train_uniform=False, upload_step=20, use_container=False, use_cuda=False, vocab_tag_size=500, vocab_token_size=10000, weight_decay=0, window='hamming', window_size=0.02, window_stride=0.01, yogi_beta=0.9, yogi_beta2=0.99, yogi_eta=0.003, yogi_tau=1e-08)

    config =  {"adam_epsilon": 1e-08, "arrival_interval": 3, "async_buffer": 10, "async_mode": False, "backbone": './resnet50.pth', "backend": 'gloo', "batch_size": 20, "bidirectional": True, "blacklist_max_len": 0.3, "blacklist_rounds": -1, "block_size": 64, "cfg_file": './utils/rcnn/cfgs/res101.yml', "checkin_period": 50, "clf_block_size": 32, "clip_bound": 0.9, "clip_threshold": 3.0, "clock_factor": 1.1624548736462095, "conf_path": '~/dataset/', "connection_timeout": 60, "cuda_device": None, "cut_off_util": 0.05, "data_cache": '', "data_dir": '/FedScale/benchmark/dataset/data/femnist', "data_map_file": '/FedScale/benchmark/dataset/data/femnist/client_data_mapping/train.csv', "data_set": 'femnist', "decay_factor": 0.98, "decay_round": 10, "device_avail_file": '/FedScale/benchmark/dataset/data/device_info/client_behave_trace', "device_conf_file": '/FedScale/benchmark/dataset/data/device_info/client_device_capacity', "dump_epoch": 10000000000.0, "embedding_file": 'glove.840B.300d.txt', "engine": 'pytorch', "epsilon": 0.9, "eval_interval": 5, "executor_configs": '10.0.1.8:[1]', "experiment_mode": 'simulation', "exploration_alpha": 0.3, "exploration_decay": 0.98, "exploration_factor": 0.9, "exploration_min": 0.3, "filter_less": 21, "filter_more": 1000000000000000.0, "finetune": False, "gamma": 0.9, "gradient_policy": 'yogi', "hidden_layers": 7, "hidden_size": 256, "input_dim": 0, "job_name": 'femnist', "labels_path": 'labels.json', "learning_rate": 0.05, "line_by_line": False, "local_steps": 20, "log_path": '/FedScale/aggr_log', "loss_decay": 0.2, "malicious_factor": 4, "memory_capacity": 2000, "min_learning_rate":5e-05, "mlm": False, "mlm_probability":0.15, "model": 'shufflenet_v2_x2_0', "model_size": 65536, "model_zoo":'torchcv', "n_actions":2, "n_states": 4, "noise_dir": None, "noise_factor": 0.1, "noise_max": 0.5, "noise_min": 0.0, "noise_prob": 0.4, "num_class": 62, "num_classes": 35, "num_executors": 1, "num_loaders": 2, "num_participants": 2, "output_dim": 0, "overcommitment": 1.3, "overwrite_cache": False, "pacer_delta": 5, "pacer_step": 20, "proxy_mu": 0.1, "ps_ip": '10.0.1.6', "ps_port": '29501', "rnn_type": 'lstm', "round_penalty": 2.0, "round_threshold": 30, "rounds": 10, "sample_mode": 'random', "sample_rate": 16000, "sample_seed": 233, "sample_window": 5.0, "spec_augment": False, "speed_volume_perturb": False, "target_delta": 0.0001, "target_replace_iter": 15, "task": 'cv', "test_bsz": 20, "test_manifest": 'data/test_manifest.csv', "test_output_dir": './logs/server', "test_ratio": 1.0, "test_size_file": '', "this_rank": 0, 
    "time_stamp": '0909_000000', "train_manifest": 'data/train_manifest.csv', "train_size_file": '', "train_uniform": False, "upload_step": 20, "use_cuda": False, "vocab_tag_size": 500, "vocab_token_size": 10000, "weight_decay": 0, "window": 'hamming', "window_size": 0.02, "window_stride": 0.01, "yogi_beta": 0.9, "yogi_beta2": 0.99, "yogi_eta": 0.003, "yogi_tau": 1e-08}
    msg['data'] = config
    msg = json.dumps(msg)
    send_socket.sendall(msg.encode('utf-8'))
    # while True:
    #     time.sleep(0.1)
    #     try:
    #         data = send_socket.recv(4096)
    #     except socket.timeout:
    #         continue
    #     print(data)
    break

    