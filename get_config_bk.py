import configparser as cp


class Train(object):
    """docstring for ."""

    def __init__(self, config_file):
        self.general = {}
        self.indexes = {}
        self.pipeline = {}
        self.net = {}
        self.read_config(config_file)

    def read_config(self, config_file):
        p = cp.ConfigParser()
        p.read(config_file)
        tmp = {j: p[i][j] for i in p for j in p[i]}
        num_bands = int(tmp['num_bands'])
        N = int(tmp['n'])
        # GENERAL
        self.general['name'] = tmp['name']
        self.general['path_dali'] = tmp['path_dali']
        self.general['path_output'] = tmp['path_output']
        self.general['net_type'] = tmp['net_type']
        # INDEXES
        self.indexes['step'] = int(tmp['step'])
        self.indexes['chunk_size'] = int(tmp['chunk_size'])
        self.indexes['N'] = N
        # NET
        self.net['lr'] = float(tmp['lr'])
        self.net['act_g'] = tmp['act_g']
        self.net['act_b'] = tmp['act_b']
        self.net['act_last'] = tmp['act_last']
        self.net['emb_type'] = tmp['emb_type']
        self.net['film_type'] = tmp['film_type']
        self.net['input_shape'] = (num_bands, N, 1)
        self.net['num_cond'] = int(tmp['num_cond'])
        self.net['c_type'] = tmp['c_type']
        # PIPELINE
        self.pipeline['dim'] = (N, num_bands)
        self.pipeline['batch_size'] = int(tmp['batch_size'])
        self.pipeline['num_cond'] = int(tmp['num_cond'])
        self.pipeline['n_batch'] = int(tmp['n_batch'])
        self.pipeline['N'] = N
        self.pipeline['emb_type'] = tmp['emb_type']
        self.pipeline['net_type'] = tmp['net_type']
        self.pipeline['c_type'] = tmp['c_type']


class Test(object):
    """docstring for ."""

    def __init__(self, config_file):
        self.general = {}
        self.net = {'n': 128, 'overlap': 0, 'num_bands': 512, 'num_phone': 40}
        self.read_config(config_file)

    def read_config(self, config_file):
        p = cp.ConfigParser()
        p.read(config_file)
        tmp = {j: p[i][j] for i in p for j in p[i]}
        self.general['path_dali'] = tmp['path_dali']
        self.general['path_test'] = tmp['path_test']
        self.general['path_model'] = tmp['path_model']
        self.general['path_results'] = tmp['path_results']
        self.net['net_type'] = tmp['net_type']
        self.net['emb_type'] = tmp['emb_type']
        self.net['input_type'] = tmp['input_type'].split('_')
