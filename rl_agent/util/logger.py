import pickle

import subprocess
import socket
import datetime

class Log():
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

        self._is_initialized = False
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
        self.params[name] = value
        self.params_info[name] = info

    def add_data_item(self, name, info=''):
        self._is_initialized = True
        if not self._recording_started:
            self.data[name] = []
            self.data_info[name] = info
        else:
            raise ValueError('Cant add data items after calling append')


    def append(self, episode, step, total_step, **data):

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
        arr = self.data[key]
        item = self.data[key][-1] if len(arr) > 0 else None
        episode = self.episodes[-1]
        step = self.steps[-1]
        total_step = self.total_steps[-1]
        return item, episode, step, total_step

class Logger():
    def __init__(self):

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
