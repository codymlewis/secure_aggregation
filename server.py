class Server:

    def __init__(self, network, params, R=2**16 - 1):
        network.setup()
        self.network = network
        self.R = R

    def step(self):
        self.network.begin("grads")
        grads, count = self.network(self.R)
        self.network.send_grads(grads / count)

    def analysis(self):
        self.network.begin("loss")
        loss, count = self.network(self.R)
        return loss[0] / count
