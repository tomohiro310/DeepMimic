import learning.nets.fc_2layers_256units as fc_2layers_256units

def build_net(net_name, input_tfs, reuse=False):
    net = None

    if (net_name == fc_2layers_256units.NAME):
        net = fc_2layers_256units.build_net(input_tfs, reuse)
    else:
        assert False, 'Unsupported net: ' + net_name
    
    return net
