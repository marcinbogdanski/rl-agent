import numpy as np
import collections

class NeuralNetwork2:
    def __init__(self, shape):
        '''Simplest possible 2-layer perceptron

        Arg:
            shape - 3-tuple: (nb_inputs, nb_hidden, nb_outputs)
        '''
        self.nb_inputs = shape[0]
        self.nb_hidden = shape[1]
        self.nb_outputs = shape[2]

        if len(shape) != 3:
            raise ValueError('This implementation supports only 2 layer NN')
        
        self.weights = [None, None]
        self.biases = [None, None]

        # hidden layer
        self.weights_hidden = np.random.randn(shape[0], shape[1])# * 0.1
        self.biases_hidden = np.random.randn(1, shape[1])# * 0.1

        # output layer
        self.weights_output = np.random.randn(shape[1], shape[2])# * 0.1
        self.biases_output = np.random.randn(1, shape[2])

        # # hidden layer
        # l_0 = np.sqrt(6 / (shape[0] + shape[1]))
        # self.weights_hidden = np.random.uniform(-l_0, l_0, (shape[0], shape[1]))
        # self.biases_hidden = np.ones([1, shape[1]]) * 0.1

        # # output layer
        # l_1 = np.sqrt(6 / (shape[1] + shape[2]))
        # self.weights_output = np.random.uniform(-l_0, l_0, (shape[1], shape[2]))
        # self.biases_output = np.ones([1, shape[2]]) * 0.1

        self.grad_log = []
        self.w_abs_max = collections.deque(maxlen=5000)
        self.b_abs_max = collections.deque(maxlen=5000)
        self.w2_abs_max = collections.deque(maxlen=5000)
        self.b2_abs_max = collections.deque(maxlen=5000)

    def fun_linear(self, x, deriv=False):
        if deriv:
            return 1
        return x

    def fun_sigmoid(self, x, deriv=False):
        if deriv:
            return np.multiply(self.fun_sigmoid(x), (1 - self.fun_sigmoid(x)))
        return 1 / (1 + np.exp(-x))

    def fun_relu(self, x, deriv=False):
        if deriv:
            1. * (x >= 0)
        return x * (x >= 0)



    def forward(self, data):

        # hidden layer
        temp = np.dot(data, self.weights_hidden)
        inputs_hidden = np.add(temp, self.biases_hidden)
        outputs_hidden = self.fun_sigmoid(inputs_hidden)

        # output layer
        temp = np.dot(outputs_hidden, self.weights_output)
        inputs_output =  np.add(temp, self.biases_output)
        outputs_output = self.fun_linear(inputs_output)

        return outputs_output

    def backward(self, data, labels):
        
        # forward hidden layer
        temp = np.dot(data, self.weights_hidden)
        inputs_hidden = np.add(temp, self.biases_hidden)
        outputs_hidden = self.fun_sigmoid(inputs_hidden)

        # forward output layer
        temp = np.dot(outputs_hidden, self.weights_output)
        inputs_output =  np.add(temp, self.biases_output)
        outputs_output = self.fun_linear(inputs_output)

        temp = (outputs_output - labels)
        error_term_out = temp * self.fun_linear(inputs_output, deriv=True)

        d_weights_output = np.dot(outputs_hidden.T, error_term_out)
        d_biases_output = error_term_out


        temp = np.dot(error_term_out, self.weights_output.T)
        error_term_hid = temp * self.fun_sigmoid(inputs_hidden, deriv=True)

        d_weights_hidden = np.dot(data.T, error_term_hid)
        d_biases_hidden = error_term_hid


        res_b = [d_biases_hidden, d_biases_output]
        res_w = [d_weights_hidden, d_weights_output] 

        # sum biases in case data was multi-row
        res_b[0] = np.sum(res_b[0], axis=0, keepdims=True)
        res_b[1] = np.sum(res_b[1], axis=0, keepdims=True)

        #grad = np.sum(res_b[0]) + np.sum(res_b[1]) + \
        #       np.sum(res_w[0]) + np.sum(res_w[1])
        # grad = np.max(np.abs(res_w[0]))
        
        cc = 10
        # res_w[0] = np.clip(res_w[0], -cc, cc)
        # res_b[0] = np.clip(res_b[0], -cc, cc)
        # res_w[1] = np.clip(res_w[1], -cc, cc)
        # res_b[1] = np.clip(res_b[1], -cc, cc)

        self.w_abs_max.append(np.max(np.abs(res_w[0])))
        self.b_abs_max.append(np.max(np.abs(res_b[0])))
        self.w2_abs_max.append(np.max(np.abs(res_w[1])))
        self.b2_abs_max.append(np.max(np.abs(res_b[1])))

        return res_b, res_w

    def train_batch(self, inputs, targets, eta):
        assert isinstance(inputs, np.ndarray)
        assert isinstance(targets, np.ndarray)
        assert inputs.ndim == 2
        assert targets.ndim == 2
        assert inputs.shape[1] == self.nb_inputs
        assert targets.shape[1] == self.nb_outputs
        assert len(inputs) == len(targets)
        assert len(inputs) != 0
        
        del_b, del_w = self.backward(inputs, targets)

        self.weights_hidden += -eta / len(inputs) * del_w[0]
        self.weights_output += -eta / len(inputs) * del_w[1]
        self.biases_hidden += -eta / len(inputs) * del_b[0]
        self.biases_output += -eta / len(inputs) * del_b[1]

    def evaluate(self, data):
        total_error = 0
        count = 0
        for x, y in data:
            out = self.forward(x)
            total_error += np.sum(np.square(out - y))
            if np.argmax(out) == np.argmax(y):
                count += 1

        return total_error, count