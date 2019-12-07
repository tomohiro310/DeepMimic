import learning.nets.fc_2layers_2048units as fc_2layers_2048units

def build_net(net_name, input_tfs, reuse=False):
    net = None

    if (net_name == fc_2layers_2048units.NAME):
        net = fc_2layers_2048units.build_net(input_tfs, reuse)
    else:
        assert False, 'Unsupported net: ' + net_name
    
    return net
