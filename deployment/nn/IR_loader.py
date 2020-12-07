from openvino.inference_engine import IEPlugin, IENetwork

class InferenceNetwork ():
    @staticmethod
    def load (model, weights):
        try:
            # load intermediate representation
            net = IENetwork(model=model, weights=weights)
            # initialize 
            plugin = IEPlugin(device='MYRIAD')
            
            #load model to plugin
            execNet = plugin.load(network=net, num_requests=2)
            
            return net, execNet
        except Exception as e:
            raise e