# -*- coding: utf-8 -*-
import collections
import gc
import pickle
from argparse import Namespace

import torch

import fedscale.core.channels.job_api_pb2 as job_api_pb2
from fedscale.core import commons
from fedscale.core.channels.channel_context import ClientConnections
from fedscale.core.execution.client import Client
from fedscale.core.execution.data_processor import collate, voice_collate_fn
from fedscale.core.execution.rlclient import RLClient
from fedscale.core.logger.execution import *
import fedscale.core.config_parser as parser

CONTAINER_IP = "0.0.0.0"
CONTAINER_PORT = 32000

class Executor(object):
    """Abstract class for FedScale executor.

    Args:
        args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

    """
    def __init__(self, args):

        self.args = args
        self.device = args.cuda_device if args.use_cuda else torch.device(
            'cpu')
        self.num_executors = args.num_executors
        # ======== env information ========
        self.this_rank = args.this_rank
        self.executor_id = str(self.this_rank)

        # ======== model and data ========
        self.model = self.training_sets = self.test_dataset = None
        self.temp_model_path = os.path.join(
            logDir, 'model_'+str(args.this_rank)+'.pth.tar')

        # ======== channels ========
        self.aggregator_communicator = ClientConnections(
            args.ps_ip, args.ps_port)

        # ======== runtime information ========
        self.collate_fn = None
        self.task = args.task
        self.round = 0
        self.start_run_time = time.time()
        self.received_stop_request = False
        self.event_queue = collections.deque()

        super(Executor, self).__init__()

    def setup_env(self):
        """Set up experiments environment
        """
        logging.info(f"(EXECUTOR:{self.this_rank}) is setting up environ ...")
        self.setup_seed(seed=1)

    def setup_communication(self):
        """Set up grpc connection
        """
        self.init_control_communication()
        self.init_data_communication()

    def setup_seed(self, seed=1):
        """Set random seed for reproducibility

        Args:
            seed (int): random seed

        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def init_control_communication(self):
        """Create communication channel between coordinator and executor.
        This channel serves control messages.
        """
        self.aggregator_communicator.connect_to_server()

    def init_data_communication(self):
        """In charge of jumbo data traffics (e.g., fetch training result)
        """
        pass

    def init_model(self):
        """Get the model architecture used in training

        Returns: 
            PyTorch or TensorFlow module: Based on the executor's machine learning framework, initialize and return the model for training
        
        """
        assert self.args.engine == commons.PYTORCH, "Please override this function to define non-PyTorch models"
        model = init_model()
        model = model.to(device=self.device)
        return model

    def init_data(self):
        """Return the training and testing dataset

        Returns:
            Tuple of DataPartitioner class: The partioned dataset class for training and testing

        """
        train_dataset, test_dataset = init_dataset()
        if self.task == "rl":
            return train_dataset, test_dataset
        # load data partitioner (entire_train_data)
        logging.info("Data partitioner starts ...")

        training_sets = DataPartitioner(
            data=train_dataset, args=self.args, numOfClass=self.args.num_class)
        training_sets.partition_data_helper(
            num_clients=self.args.num_participants, data_map_file=self.args.data_map_file)

        testing_sets = DataPartitioner(
            data=test_dataset, args=self.args, numOfClass=self.args.num_class, isTest=True)
        testing_sets.partition_data_helper(num_clients=self.num_executors)

        logging.info("Data partitioner completes ...")

        if self.task == 'nlp':
            self.collate_fn = collate
        elif self.task == 'voice':
            self.collate_fn = voice_collate_fn

        return training_sets, testing_sets

    def run(self):
        """Start running the executor by setting up execution and communication environment, and monitoring the grpc message.
        """
        self.setup_env()
        self.model = self.init_model()
        self.training_sets, self.testing_sets = self.init_data()
        self.setup_communication()
        self.event_monitor()

    def dispatch_worker_events(self, request):
        """Add new events to worker queues
        
        Args:
            request (string): Add grpc request from server (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.
        
        """
        self.event_queue.append(request)

    def deserialize_response(self, responses):
        """Deserialize the response from server

        Args:
            responses (byte stream): Serialized response from server.

        Returns:
            ServerResponse defined at job_api.proto: The deserialized response object from server.
        
        """
        return pickle.loads(responses)

    def serialize_response(self, responses):
        """Serialize the response to send to server upon assigned job completion

        Args:
            responses (string, bool, or bytes): Client responses after job completion.

        Returns:
            bytes stream: The serialized response object to server.
        
        """
        return pickle.dumps(responses)

    def UpdateModel(self, config):
        """Receive the broadcasted global model for current round

        Args:
            config (PyTorch or TensorFlow model): The broadcasted global model config
        
        """
        self.update_model_handler(model=config)

    def Train(self, config):
        """Load train config and data to start training on that client

        Args:
            config (dictionary): The client training config.

        Returns:     
            tuple (int, dictionary): The client id and train result

        """
        client_id, train_config = config['client_id'], config['task_config']

        model = None
        if 'model' in train_config and train_config['model'] is not None:
            model = train_config['model']

        client_conf = self.override_conf(train_config)
        train_res = self.training_handler(
            clientId=client_id, conf=client_conf, model=model)

        # Report execution completion meta information
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
            job_api_pb2.CompleteRequest(
                client_id=str(client_id), executor_id=self.executor_id,
                event=commons.CLIENT_TRAIN, status=True, msg=None,
                meta_result=None, data_result=None
            )
        )
        self.dispatch_worker_events(response)

        return client_id, train_res

    def Test(self, config):
        """Model Testing. By default, we test the accuracy on all data of clients in the test group
        
        Args:
            config (dictionary): The client testing config.
        
        """
        test_res = self.testing_handler(args=self.args)
        test_res = {'executorId': self.this_rank, 'results': test_res}

        # Report execution completion information
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
            job_api_pb2.CompleteRequest(
                client_id=self.executor_id, executor_id=self.executor_id,
                event=commons.MODEL_TEST, status=True, msg=None,
                meta_result=None, data_result=self.serialize_response(test_res)
            )
        )
        self.dispatch_worker_events(response)

    def Stop(self):
        """Stop the current executor
        """
        self.aggregator_communicator.close_sever_connection()
        self.received_stop_request = True

    def report_executor_info_handler(self):
        """Return the statistics of training dataset

        Returns:
            int: Return the statistics of training dataset, in simulation return the number of clients

        """
        return self.training_sets.getSize()

    def update_model_handler(self, model):
        """Update the model copy on this executor

        Args:
            config (PyTorch or TensorFlow model): The broadcasted global model

        """
        self.model = model
        self.round += 1

        # Dump latest model to disk
        with open(self.temp_model_path, 'wb') as model_out:
            pickle.dump(self.model, model_out)

    def load_global_model(self):
        """ Load last global model

        Returns:
            PyTorch or TensorFlow model: The lastest global model

        """
        with open(self.temp_model_path, 'rb') as model_in:
            model = pickle.load(model_in)
        return model

    def override_conf(self, config):
        """ Override the variable arguments for different client

        Args:
            config (dictionary): The client runtime config.

        Returns:
            dictionary: Variable arguments for client runtime config.

        """
        default_conf = vars(self.args).copy()

        for key in config:
            default_conf[key] = config[key]

        return Namespace(**default_conf)

    def get_client_trainer(self, conf):
        """A abstract base class for client with training handler, developer can redefine to this function to customize the client training:

        Args:
            config (dictionary): The client runtime config.

        Returns:
            Client: A abstract base client class with runtime config conf.

        """
        return Client(conf)

    def training_handler(self, clientId, conf, model=None):
        """Train model given client id
        
        Args:
            clientId (int): The client id.
            conf (dictionary): The client runtime config.

        Returns:
            dictionary: The train result
        
        """
        # load last global model
        client_model = self.load_global_model() if model is None else model

        conf.clientId, conf.device = clientId, self.device
        conf.tokenizer = tokenizer
        if self.args.task == "rl":
            client_data = self.training_sets
            client = RLClient(conf)
            train_res = client.train(
                client_data=client_data, model=client_model, conf=conf)
        else:
            client_data = select_dataset(clientId, self.training_sets,
                                         batch_size=conf.batch_size, args=self.args,
                                         collate_fn=self.collate_fn
                                         )

            client = self.get_client_trainer(conf)
            train_res = client.train(
                client_data=client_data, model=client_model, conf=conf)

        return train_res

    def testing_handler(self, args):
        """Test model
        
        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

        Returns:
            dictionary: The test result

        """
        evalStart = time.time()
        device = self.device
        model = self.load_global_model()
        if self.task == 'rl':
            client = RLClient(args)
            test_res = client.test(args, self.this_rank, model, device=device)
            _, _, _, testResults = test_res
        else:
            data_loader = select_dataset(self.this_rank, self.testing_sets,
                                         batch_size=args.test_bsz, args=args,
                                         isTest=True, collate_fn=self.collate_fn
                                         )

            if self.task == 'voice':
                criterion = CTCLoss(reduction='mean').to(device=device)
            else:
                criterion = torch.nn.CrossEntropyLoss().to(device=device)

            if self.args.engine == commons.PYTORCH:
                test_res = test_model(self.this_rank, model, data_loader,
                                      device=device, criterion=criterion, tokenizer=tokenizer)
            else:
                raise Exception(f"Need customized implementation for model testing in {self.args.engine} engine")

            test_loss, acc, acc_5, testResults = test_res
            logging.info("After aggregation round {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                         .format(self.round, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), test_loss, acc*100., acc_5*100.))

        gc.collect()

        return testResults

    def client_register(self):
        """Register the executor information to the aggregator
        """
        start_time = time.time()
        while time.time() - start_time < 180:
            try:
                response = self.aggregator_communicator.stub.CLIENT_REGISTER(
                    job_api_pb2.RegisterRequest(
                        client_id=self.executor_id,
                        executor_id=self.executor_id,
                        executor_info=self.serialize_response(
                            self.report_executor_info_handler())
                    )
                )
                self.dispatch_worker_events(response)
                break
            except Exception as e:
                logging.warning(f"Failed to connect to aggregator {e}. Will retry in 5 sec.")
                time.sleep(5)

    def client_ping(self):
        """Ping the aggregator for new task
        """
        response = self.aggregator_communicator.stub.CLIENT_PING(job_api_pb2.PingRequest(
            client_id=self.executor_id,
            executor_id=self.executor_id
        ))
        self.dispatch_worker_events(response)

    def event_monitor(self):
        """Activate event handler once receiving new message
        """
        logging.info("Start monitoring events ...")
        self.client_register()

        while self.received_stop_request == False:
            if len(self.event_queue) > 0:
                request = self.event_queue.popleft()
                current_event = request.event

                if current_event == commons.CLIENT_TRAIN:
                    train_config = self.deserialize_response(request.meta)
                    train_model = self.deserialize_response(request.data)
                    train_config['model'] = train_model
                    train_config['client_id'] = int(train_config['client_id'])
                    client_id, train_res = self.Train(train_config)

                    # Upload model updates
                    future_call = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION.future(
                        job_api_pb2.CompleteRequest(client_id=str(client_id), executor_id=self.executor_id,
                                                    event=commons.UPLOAD_MODEL, status=True, msg=None,
                                                    meta_result=None, data_result=self.serialize_response(train_res)
                                                    ))
                    future_call.add_done_callback(lambda _response: self.dispatch_worker_events(_response.result()))

                elif current_event == commons.MODEL_TEST:
                    self.Test(self.deserialize_response(request.meta))

                elif current_event == commons.UPDATE_MODEL:
                    broadcast_config = self.deserialize_response(request.data)
                    self.UpdateModel(broadcast_config)

                elif current_event == commons.SHUT_DOWN:
                    self.Stop()

                elif current_event == commons.DUMMY_EVENT:
                    pass
            else:
                time.sleep(1)
                self.client_ping()

def exec_container_init():
    """ Initialization needed if executor is running inside a container
    """
    # This is almost like aggr_container_init(), uniquely defined here to allow for future executor-specific init
    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.bind((CONTAINER_IP, CONTAINER_PORT))
    listen_socket.settimeout(1)
    listen_socket.listen(5)
    logging.info("Executor waiting to initialize")
    while True:
        # avoid busy waiting
        time.sleep(0.1)
        try:
            incoming_socket, addr = listen_socket.accept()
        except socket.timeout:
            continue
        message_chunks = []
        while True:
            time.sleep(0.1)
            try:
                msg = incoming_socket.recv(4096)
            except socket.timeout:
                continue
            if not msg:
                break
            message_chunks.append(msg)
        message_bytes = b''.join(message_chunks)
        message_str = message_bytes.decode('utf-8')
        incoming_socket.close()
        try:
            msg = json.loads(message_str)
        except json.JSONDecodeError:
            logging.info("Error decoding init message!")
            listen_socket.close()
            exit(1)
        if msg['type'] == 'initialize':
            logging.info("Executor init success!")
            new_args = msg['data']
            # print(args)
            listen_socket.close()
            return new_args


if __name__ == "__main__":
    if parser.args.use_container == True:
        new_args = exec_container_init()
        # Update arguments globally
        for key in new_args:
            args_dict = vars(parser.args)
            assert(key in args_dict)
            args_dict[key] = new_args[key]
    # print("executor args", args)
    # config =   {"adam_epsilon": 1e-8, "arrival_interval": 3, "async_buffer": 10, "async_mode": False, "backbone": './resnet50.pth', "backend": 'gloo', "batch_size": 20, "bidirectional": True, "blacklist_max_len": 0.3, "blacklist_rounds": -1, "block_size": 64, "cfg_file": './utils/rcnn/cfgs/res101.yml', "checkin_period": 50, "clf_block_size": 32, "clip_bound": 0.9, "clip_threshold": 3.0, "clock_factor": 1.1624548736462095, "conf_path": '~/dataset/', "connection_timeout": 60, "cuda_device": None, "cut_off_util": 0.05, "data_cache": '', "data_dir": '/users/yilegu/fedscale_k8s/FedScale/benchmark/dataset/data/femnist', "data_map_file": '/users/yilegu/fedscale_k8s/FedScale/benchmark/dataset/data/femnist/client_data_mapping/train.csv', "data_set": 'femnist', "decay_factor": 0.98, "decay_round": 10, "device_avail_file": '/users/yilegu/fedscale_k8s/FedScale/benchmark/dataset/data/device_info/client_behave_trace', "device_conf_file": '/users/yilegu/fedscale_k8s/FedScale/benchmark/dataset/data/device_info/client_device_capacity', "dump_epoch": 10000000000.0, "embedding_file": 'glove.840B.300d.txt', "engine": 'pytorch', "epsilon": 0.9, "eval_interval": 30, "executor_configs": 'localhost:[1]', "experiment_mode": 'simulation', "exploration_alpha": 0.3, "exploration_decay": 0.98, "exploration_factor": 0.9, "exploration_min": 0.3, "filter_less": 21, "filter_more": 1000000000000000.0, "finetune": False, "gamma": 0.9, "gradient_policy": 'yogi', "hidden_layers": 7, "hidden_size": 256, "input_dim": 0, "job_name": 'femnist', "labels_path": 'labels.json', "learning_rate": 0.05, "line_by_line": False, "local_steps": 20, "log_path": '/users/yilegu/fedscale_k8s/FedScale/aggr_log', "loss_decay": 0.2, "malicious_factor": 4, "memory_capacity": 2000, "min_learning_rate":5e-05, "mlm": False, "mlm_probability":0.15, "model": 'shufflenet_v2_x2_0', "model_size": 65536, "model_zoo":'torchcv', "n_actions":2, "n_states": 4, "noise_dir": None, "noise_factor": 0.1, "noise_max": 0.5, "noise_min": 0.0, "noise_prob": 0.4, "num_class": 62, "num_classes": 35, "num_executors": 1, "num_loaders": 2, "num_participants": 5, "output_dim": 0, "overcommitment": 1.3, "overwrite_cache": False, "pacer_delta": 5, "pacer_step": 20, "proxy_mu": 0.1, "ps_ip": 'localhost', "ps_port": '29501', "rnn_type": 'lstm', "round_penalty": 2.0, "round_threshold": 30, "rounds": 500, "sample_mode": 'random', "sample_rate": 16000, "sample_seed": 233, "sample_window": 5.0, "spec_augment": False, "speed_volume_perturb": False, "target_delta": 0.0001, "target_replace_iter": 15, "task": 'cv', "test_bsz": 20, "test_manifest": 'data/test_manifest.csv', "test_output_dir": './logs/server', "test_ratio": 1.0, "test_size_file": '', "this_rank": 1, 
    # "time_stamp": '0909_000000', "train_manifest": 'data/train_manifest.csv', "train_size_file": '', "train_uniform": False, "upload_step": 20, "use_cuda": False, "vocab_tag_size": 500, "vocab_token_size": 10000, "weight_decay": 0, "window": 'hamming', "window_size": 0.02, "window_stride": 0.01, "yogi_beta": 0.9, "yogi_beta2": 0.99, "yogi_eta": 0.003, "yogi_tau": 1e-08}
    # args = commons.Config(config)
    print(vars(parser.args))
    executor = Executor(parser.args)
    executor.run()
