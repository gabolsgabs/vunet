import configparser as cp

FLOAT_TYPE = ['lr']
INT_TYPE = ['n', 'num_bands', 'step', 'chunk_size', 'num_cond', 'n_batch',
            'batch_size']


def update_param(orig, new_param):
    for i in orig:
        if i.lower() in new_param:
            orig[i] = new_param[i.lower()]
    return orig


def read_config(config_file):
    p = cp.ConfigParser()
    p.read(config_file)
    output = {}
    for i in p:
        output.setdefault(i, {})
        for j in p[i]:
            if j in INT_TYPE:
                output[i][j] = int(p[i][j])
            elif j in FLOAT_TYPE:
                output[i][j] = float(p[i][j])
            else:
                output[i][j] = p[i][j]
    return output


class Train(object):
    """docstring for ."""

    def __init__(self, config_file):
        self.general = {'path_dali': None, 'path_output': None, 'name': None}
        self.pipeline = {'chunk_size': None, 'step': None, 'batch_size': None,
                         'n_batch': None}
        self.net = {
            'N': 128, 'num_bands': 512, 'lr': 0.001, 'act_g': 'linear',
            'act_b': 'linear', 'act_last': 'sigmoid', 'net_type': None,
            'emb_type': None, 'film_type': None, 'num_cond': None,
            'cond_input': None,
        }
        self.get_config(config_file)

    def get_config(self, config_file):
        config = read_config(config_file)
        # GENERAL
        self.general = update_param(self.general, config['GENERAL'])
        # NET
        self.net = update_param(self.net, config['NET'])
        self.net = update_param(self.net, config['CONDITIONS'])
        self.net['input_shape'] = (self.net['num_bands'], self.net['N'], 1)
        # PIPELINE
        self.pipeline = update_param(self.pipeline, config['PIPELINE'])

        # NEEDED PARAMS
        self.pipeline['dim'] = (self.net['N'], self.net['num_bands'])
        self.pipeline['num_cond'] = self.net['num_cond']
        self.pipeline['N'] = self.net['N']
        self.pipeline['emb_type'] = self.net['emb_type']
        self.pipeline['net_type'] = self.net['net_type']
        self.pipeline['cond_input'] = self.net['cond_input']
        if self.pipeline['cond_input'] in ['binary', 'ponderate', 'energy']:
            self.pipeline['dim_cond'] = (self.net['num_cond'], )
            self.net['input_shape_cond'] = (self.net['num_cond'],)
        elif self.pipeline['cond_input'] in ['autopool']:
            self.pipeline['dim_cond'] = (self.net['N'], self.net['num_cond'])
            self.net['input_shape_cond'] = (self.net['num_cond'], self.net['N'])


class Test(object):
    """docstring for ."""

    def __init__(self, config_file):
        self.general = {'path_dali': None, 'path_model': None,
                        'path_results': None}
        self.get_config(config_file)

    def get_config(self, config_file):
        config = read_config(config_file)
        self.general = update_param(self.general, config['GENERAL'])
