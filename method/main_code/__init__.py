from method.main_code.network import CGCCE


def get_segmentation_model(name):
    if name == 'CGCCE':
        net = CGCCE()
    else:
        raise NotImplementedError
    return net
