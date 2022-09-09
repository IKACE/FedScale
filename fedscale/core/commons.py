
# Define Basic Experiment Setup
SIMULATION_MODE = 'simulation'
DEPLOYMENT_MODE = 'deployment'
TENSORFLOW = 'tensorflow'
PYTORCH = 'pytorch'

# Define Basic FL Events
UPDATE_MODEL = 'update_model'
MODEL_TEST = 'model_test'
SHUT_DOWN = 'terminate_executor'
START_ROUND = 'start_round'
CLIENT_CONNECT = 'client_connect'
CLIENT_TRAIN = 'client_train'
DUMMY_EVENT = 'dummy_event'
UPLOAD_MODEL = 'upload_model'

# PLACEHOLD
DUMMY_RESPONSE = 'N'


class Config(object):
    def __init__(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])