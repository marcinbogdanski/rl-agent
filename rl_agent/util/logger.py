import pickle

import subprocess
import socket
import datetime

class Log_old():
    """Log containing header and multiple user-defined series of data

    One log should be used for each module. E.g. one or more for agent, one for 
    DQN memory, one for Q-approximator, one for Policy etc.

    Each log must be initialized first by calling:
        * add_param() to define header info
        * add_data_item() to define data items logged over lifetime of the Log
        * all add_param() and add_data_item() calls must happen before actuall
          logging starts

    Example:
        log = Log('DQN Memory Log')
        log.add_param('max_len', 100000)   # header info saved once
        log.add_param('enable_pmr', False)
        ...
        log.add_data_item('curr_size')  # define data to be logged
        log.add_data_item('state')      # these will have to be supplied
        log.add_data_item('action')     # every time data is passed to logger
        ...
        # actually log stuff, persumably once every couple time-steps
        log.append(curr_size=10, state=[1, 2], action=2)
    """    

    def __init__(self, name, desc=''):
        self.name = name
        self.description = desc
        
        self.params = {}
        self.params_info = {}

        self.episodes = []
        self.steps = []
        self.total_steps = []
        
        self.data = {}
        self.data_info = {}

        # set to True first time set_data_item() is called
        self._is_initialized = False

        # set to True first time append() is called, after that any further
        # calls to add_param() and add_data_item() will raise exception
        self._recording_started = False

    @property
    def is_initialized(self):
        return self._is_initialized

    def __str__(self):
        res = ''
        res += 'Log:\n'
        res += '  name: ' + self.name + '\n'
        res += '  desc: ' + self.description + '\n'
        
        if len(self.episodes) == 0:
            res += '  episodes: ' + str(len(self.episodes)) + '\n'
        else:
            res += '  episodes: ' + str(len(self.episodes)) + '[' + str(self.episodes[0]) + '-' + str(self.episodes[-1]) + ']\n'
        
        if len(self.steps) == 0:
            res += '  steps: ' + str(len(self.steps)) + '\n'
        else:
            res += '  steps: ' + str(len(self.steps)) + '[' + str(self.steps[0]) + '-' + str(self.steps[-1]) + ']\n'
        
        if len(self.total_steps) == 0:
            res += '  total_steps: ' + str(len(self.total_steps)) + '\n'
        else:
            res += '  total_steps: ' + str(len(self.total_steps)) + '[' + str(self.total_steps[0]) + '-' + str(self.total_steps[-1]) + ']\n'
        
        res += '  params:' + '\n'
        for key, val in self.params.items():
            res += '    ' + key + ': ' + str(val) + '    ' \
                   + self.params_info[key] + '\n'
        
        res += '  data items:\n'
        for key, val in self.data.items():
            res += '    ' + key + ': ' + self.data_info[key] + '\n'
        
        return res


    def add_param(self, name, value, info=''):
        """Add header info, e.g. module configuration items."""
        self.params[name] = value
        self.params_info[name] = info

    def add_data_item(self, name, info=''):
        """Specify data item for logging"""
        self._is_initialized = True
        if not self._recording_started:
            self.data[name] = []
            self.data_info[name] = info
        else:
            raise ValueError('Cant add data items after calling append')


    def append(self, episode, step, total_step, **data):
        """Actually log data

        Params:
            episode: current episode
            step: current step within episode
            total_step: step number from begining of time
            data (dict): log data as defined by add_data_item() calls
        """

        self._recording_started = True

        assert len(self.episodes) == len(self.steps)
        assert len(self.steps) == len(self.total_steps)

        if len(self.episodes) != 0:
            assert episode >= self.episodes[-1]

            if episode == self.episodes[-1]:
                assert step > self.steps[-1]
            assert total_step > self.total_steps[-1]


        assert set(self.data.keys()) == set(data.keys())
        for key, val in data.items():
            assert len(self.data[key]) == len(self.steps)

        self.episodes.append(episode)
        self.steps.append(step)
        self.total_steps.append(total_step)
        for key, val in data.items():
            self.data[key].append(val)

    def get_last(self, key):
        """Get most recent data item for specified key"""
        arr = self.data[key]
        item = self.data[key][-1] if len(arr) > 0 else None
        episode = self.episodes[-1]
        step = self.steps[-1]
        total_step = self.total_steps[-1]
        return item, episode, step, total_step

class Logger_old():
    def __init__(self):
        """Wrapper around multiple Log objects

        This class can save/load itself from a file.
        """

        self.datetime = str(datetime.datetime.now())  # date and time
        self.hostname = socket.gethostname()  # name of PC where script is run
        res = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
        self.git_hash = res.stdout.decode('utf-8')  # git revision if any

    def __str__(self):
        res = 'Date time: ' + self.datetime + '\n' + \
              'Hostname:' + self.hostname + '\n' + \
              'GIT Hash:' + self.git_hash
        return res

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump( self.__dict__, f )

    def load(self, filename):
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)
            self.__dict__.clear()
            self.__dict__.update(tmp_dict)




class Log():
    def __init__(self, shape, extent=None):
        self._shape = shape
        self._extent = extent

    def add(self, value):
        if value.shape != self._shape:
            raise ValueError('Shape missmatch')

class Logger():
    def __init__(self):
        self._dict = {}

    def new(self, name, log):
        self._dict[name] = log


    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump( self.__dict__, f )

    def load(self, filename):
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)
            self.__dict__.clear()
            self.__dict__.update(tmp_dict)
