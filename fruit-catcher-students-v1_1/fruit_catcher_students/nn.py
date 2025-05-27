import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, hidden_architecture, hidden_activation, output_activation):
        self.input_size = input_size
        # hidden_architecture is a tuple with the number of neurons in each hidden layer
        # e.g. (5, 2) corresponds to a neural network with 2 hidden layers in which the first has 5 neurons and the second has 2
        self.hidden_architecture = hidden_architecture
        # The activations are functions 
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        
        # Define o tamanho das camadas (entrada + escondidas + 1 saída)
        layer_sizes = [input_size] + list(hidden_architecture) + [1]
        self.weights = []

          # Cria uma matriz de pesos (com bias) entre cada par de camadas
        for i in range(len(layer_sizes) - 1):
            rows = layer_sizes[i + 1]
            cols = layer_sizes[i] + 1  # +1 para bias
            self.weights.append(np.zeros((rows, cols)))

    def compute_num_weights(self): # Implement this. Remember to account for the biases.
        total = 0
        input_size = self.input_size

        for n in self.hidden_architecture:
            total += (input_size + 1) * n  # pesos + bias incluído
            input_size = n

        # camada de saída: 1 bias + n pesos
        total += input_size + 1
        return total

    def load_weights(self, weights):
        w = np.array(weights)

        self.hidden_weights = []
        self.hidden_biases = []

        start_w = 0
        input_size = self.input_size
        for n in self.hidden_architecture:
            end_w = start_w + (input_size + 1) * n
            self.hidden_biases.append(w[start_w:start_w+n])
            self.hidden_weights.append(w[start_w+n:end_w].reshape(input_size, n))
            start_w = end_w
            input_size = n

        self.output_bias = w[start_w]
        self.output_weights = w[start_w+1:]


    def forward(self, x): # Implement this
        a = np.array(x)

        # Passa pelas camadas escondidas
        for w, b in zip(self.hidden_weights, self.hidden_biases):
            z = np.dot(a, w) + b
            a = self.hidden_activation(z)

        # Camada de saída
        z_out = np.dot(a, self.output_weights) + self.output_bias
        print("Output bruto:", z_out, "| Output final:", self.output_activation(2*z_out))
        return self.output_activation(2*z_out)


def create_network_architecture(input_size):
    hidden_fn = np.tanh
    output_fn = lambda x: -1 if x < -0.3 else (1 if x > 0.3 else 0)
    return NeuralNetwork(
        input_size=input_size,
        hidden_architecture=(10,),              
        hidden_activation=hidden_fn,
        output_activation=output_fn
    )