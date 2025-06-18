import numpy as np
import random
from copy import deepcopy

from config.constants import MAX_VELOCITY, MAX_ROTATIONAL_VELOCITY
from world.behavioral import BehavioralModel


class FlexibleNeuralNetwork:
    """
    A flexible neural network that can mutate its structure and weights.
    Supports variable topology with cross-layer connections.
    """

    def __init__(self, input_size=2, output_size=2, empty_start=True):
        self.input_size = input_size
        self.output_size = output_size

        # Network structure: list of layers, each layer is a list of neurons
        # Each neuron is represented by its connections and bias
        self.layers = []

        # Initialize network based on empty_start parameter
        if empty_start:
            self._initialize_empty_network()
        else:
            self._initialize_basic_network()

        self.network_cost = self.calculate_network_cost()

    def _initialize_basic_network(self):
        """Initialize a basic network with input->output connections only."""
        # Input layer (no actual neurons, just placeholders)
        input_layer = [{'type': 'input', 'id': i} for i in range(self.input_size)]

        # Output layer with connections to all inputs
        output_layer = []
        for i in range(self.output_size):
            neuron = {
                'type': 'output',
                'id': f'out_{i}',
                'bias': random.uniform(-1, 1),
                'connections': []  # List of (source_layer, source_neuron, weight)
            }

            # Connect to all input neurons
            for j in range(self.input_size):
                neuron['connections'].append((0, j, random.uniform(-2, 2)))

            output_layer.append(neuron)

        self.layers = [input_layer, output_layer]

    def _initialize_empty_network(self):
        """Initialize an empty network with no connections or biases."""
        # Input layer (no actual neurons, just placeholders)
        input_layer = [{'type': 'input', 'id': i} for i in range(self.input_size)]

        # Output layer with no connections and zero bias
        output_layer = []
        for i in range(self.output_size):
            neuron = {
                'type': 'output',
                'id': f'out_{i}',
                'bias': 0.0,
                'connections': []  # Empty connections list
            }
            output_layer.append(neuron)

        self.layers = [input_layer, output_layer]

    def _remove_duplicate_connections(self):
        """Remove duplicate connections and keep only the last weight for each unique connection."""
        for layer in self.layers[1:]:  # Skip input layer
            for neuron in layer:
                if 'connections' not in neuron:
                    continue

                # Use a dictionary to track unique connections by (source_layer, source_neuron)
                unique_connections = {}

                for source_layer, source_neuron, weight in neuron['connections']:
                    connection_key = (source_layer, source_neuron)
                    # Keep the last weight encountered for this connection
                    unique_connections[connection_key] = weight

                # Rebuild connections list without duplicates
                neuron['connections'] = [
                    (source_layer, source_neuron, weight)
                    for (source_layer, source_neuron), weight in unique_connections.items()
                ]

    def _connection_exists(self, target_neuron, source_layer_idx, source_neuron_idx):
        """Check if a connection already exists between two neurons."""
        if 'connections' not in target_neuron:
            return False

        for source_layer, source_neuron, weight in target_neuron['connections']:
            if source_layer == source_layer_idx and source_neuron == source_neuron_idx:
                return True
        return False

    def forward(self, inputs):
        """
        Forward pass through the network.

        :param inputs: List or array of input values
        :return: List of output values
        """
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")

        # Store activations for each layer
        activations = [inputs]  # Input layer activations

        # Process each subsequent layer
        for layer_idx in range(1, len(self.layers)):
            layer_activations = []

            for neuron in self.layers[layer_idx]:
                if neuron['type'] == 'input':
                    continue  # Skip input neurons in hidden layers

                # Calculate weighted sum of inputs
                weighted_sum = 0.0  # Start with 0 instead of bias

                # Only add bias if neuron has connections
                if 'connections' in neuron and len(neuron['connections']) > 0:
                    weighted_sum = neuron['bias']

                    for source_layer, source_neuron, weight in neuron['connections']:
                        if source_layer < len(activations):
                            if source_neuron < len(activations[source_layer]):
                                weighted_sum += activations[source_layer][source_neuron] * weight

                # Apply activation function (tanh for bounded output)
                # If no connections and no bias applied, this will be tanh(0) = 0
                activation = np.tanh(weighted_sum)
                layer_activations.append(activation)

            activations.append(layer_activations)

        return activations[-1]  # Return output layer activations

    def mutate(self, mutation_rate=0.1):
        """
        Create a mutated copy of this network.

        :param mutation_rate: Base probability multiplied by specific mutation weights
        :return: New mutated FlexibleNeuralNetwork instance
        """
        mutated = deepcopy(self)

        # Weighted mutations (probability = mutation_rate * weight)
        # Higher weights = more likely to occur
        mutations = [
            (mutated._mutate_weights, 5.0),  # Most common - fine-tune existing
            (mutated._mutate_biases, 3.0),  # Common - adjust neuron thresholds
            (mutated._add_connection, 1.5),  # Moderate - grow connectivity
            (mutated._remove_connection, 0.8),  # Less common - reduce connectivity
            (mutated._add_neuron, 0.3),  # Rare - structural growth
            (mutated._remove_neuron, 0.1),  # Very rare - structural reduction
            (mutated._add_layer, 0.05),  # New: create a new layer (very rare)
        ]

        # Apply weighted random mutations
        for mutation_func, weight in mutations:
            if random.random() < (mutation_rate * weight):
                mutation_func()

        # Clean up any duplicate connections that might have been created
        mutated._remove_duplicate_connections()

        # Ensure the network maintains basic connectivity
        mutated._ensure_network_connectivity()

        mutated.network_cost = mutated.calculate_network_cost()

        return mutated

    def _mutate_weights(self):
        """Slightly modify existing connection weights."""
        for layer in self.layers[1:]:  # Skip input layer
            for neuron in layer:
                if 'connections' in neuron:
                    for i in range(len(neuron['connections'])):
                        if random.random() < 0.3:  # 30% chance to mutate each weight
                            source_layer, source_neuron, weight = neuron['connections'][i]
                            # Add small random change
                            new_weight = weight + random.uniform(-0.5, 0.5)
                            neuron['connections'][i] = (source_layer, source_neuron, new_weight)

    def _mutate_biases(self):
        """Slightly modify neuron biases."""
        for layer in self.layers[1:]:  # Skip input layer
            for neuron in layer:
                if 'bias' in neuron and random.random() < 0.3:
                    neuron['bias'] += random.uniform(-0.5, 0.5)

    def _add_connection(self):
        """Add a new random connection."""
        if len(self.layers) < 2:
            return

        # Find layers with neurons
        valid_target_layers = []
        for i in range(1, len(self.layers)):
            if len(self.layers[i]) > 0:
                valid_target_layers.append(i)

        if not valid_target_layers:
            return

        # Pick a random target neuron (not in input layer)
        target_layer_idx = random.choice(valid_target_layers)
        target_neuron_idx = random.randint(0, len(self.layers[target_layer_idx]) - 1)
        target_neuron = self.layers[target_layer_idx][target_neuron_idx]

        if 'connections' not in target_neuron:
            return

        # Find valid source layers (must have neurons and be before target)
        valid_source_layers = []
        for i in range(target_layer_idx):
            if len(self.layers[i]) > 0:
                valid_source_layers.append(i)

        if not valid_source_layers:
            return

        # Pick a random source (from any previous layer with neurons)
        source_layer_idx = random.choice(valid_source_layers)
        source_neuron_idx = random.randint(0, len(self.layers[source_layer_idx]) - 1)

        # Check if connection already exists using the helper method
        if self._connection_exists(target_neuron, source_layer_idx, source_neuron_idx):
            return  # Connection already exists, don't add duplicate

        # Add new connection
        new_weight = random.uniform(-2, 2)
        target_neuron['connections'].append((source_layer_idx, source_neuron_idx, new_weight))

    def _remove_connection(self):
        """Remove a random connection."""
        for layer in self.layers[1:]:
            for neuron in layer:
                if 'connections' in neuron and len(neuron['connections']) > 1:
                    if random.random() < 0.1:  # 10% chance to remove a connection
                        neuron['connections'].pop(random.randint(0, len(neuron['connections']) - 1))

    def _add_neuron(self):
        """Add a new neuron to a random hidden layer or create a new hidden layer."""
        if len(self.layers) == 2:  # Only input and output layers
            # Create a new hidden layer
            hidden_neuron = {
                'type': 'hidden',
                'id': f'hidden_{random.randint(1000, 9999)}',
                'bias': random.uniform(-1, 1),
                'connections': []
            }

            # Connect to some input neurons (avoid duplicates)
            for i in range(self.input_size):
                if random.random() < 0.7:  # 70% chance to connect to each input
                    if not self._connection_exists(hidden_neuron, 0, i):
                        hidden_neuron['connections'].append((0, i, random.uniform(-2, 2)))

            # Insert hidden layer
            self.layers.insert(1, [hidden_neuron])

            # Update output layer connections to potentially use new hidden neuron
            for neuron in self.layers[-1]:  # Output layer (now at index 2)
                if random.random() < 0.5:  # 50% chance to connect to new hidden neuron
                    if not self._connection_exists(neuron, 1, 0):
                        neuron['connections'].append((1, 0, random.uniform(-2, 2)))

        else:
            # Add neuron to existing hidden layer
            # Find hidden layers that exist
            hidden_layer_indices = []
            for i in range(1, len(self.layers) - 1):
                if i < len(self.layers):  # Safety check
                    hidden_layer_indices.append(i)

            if not hidden_layer_indices:
                return

            hidden_layer_idx = random.choice(hidden_layer_indices)
            new_neuron = {
                'type': 'hidden',
                'id': f'hidden_{random.randint(1000, 9999)}',
                'bias': random.uniform(-1, 1),
                'connections': []
            }

            # Connect to some neurons from previous layers (avoid duplicates)
            for layer_idx in range(hidden_layer_idx):
                if len(self.layers[layer_idx]) > 0:  # Only if layer has neurons
                    for neuron_idx in range(len(self.layers[layer_idx])):
                        if random.random() < 0.3:  # 30% chance to connect
                            if not self._connection_exists(new_neuron, layer_idx, neuron_idx):
                                new_neuron['connections'].append((layer_idx, neuron_idx, random.uniform(-2, 2)))

            self.layers[hidden_layer_idx].append(new_neuron)

            # Update connections from later layers to potentially connect to this new neuron
            new_neuron_idx = len(self.layers[hidden_layer_idx]) - 1
            for later_layer_idx in range(hidden_layer_idx + 1, len(self.layers)):
                if len(self.layers[later_layer_idx]) > 0:  # Only if layer has neurons
                    for neuron in self.layers[later_layer_idx]:
                        if random.random() < 0.2:  # 20% chance to connect to new neuron
                            if not self._connection_exists(neuron, hidden_layer_idx, new_neuron_idx):
                                neuron['connections'].append((hidden_layer_idx, new_neuron_idx, random.uniform(-2, 2)))

    def _remove_neuron(self):
        """Remove a random neuron from hidden layers."""
        if len(self.layers) <= 2:  # No hidden layers
            return

        # Find hidden layers that have neurons
        valid_hidden_layers = []
        for layer_idx in range(1, len(self.layers) - 1):  # Only hidden layers
            if len(self.layers[layer_idx]) > 0:
                valid_hidden_layers.append(layer_idx)

        if not valid_hidden_layers:
            return

        # Pick a random hidden layer with neurons
        layer_idx = random.choice(valid_hidden_layers)
        neuron_idx = random.randint(0, len(self.layers[layer_idx]) - 1)

        # Remove the neuron
        self.layers[layer_idx].pop(neuron_idx)

        # Remove connections to this neuron from later layers
        for later_layer_idx in range(layer_idx + 1, len(self.layers)):
            for neuron in self.layers[later_layer_idx]:
                if 'connections' in neuron:
                    neuron['connections'] = [
                        (src_layer, src_neuron, weight)
                        for src_layer, src_neuron, weight in neuron['connections']
                        if not (src_layer == layer_idx and src_neuron == neuron_idx)
                    ]

        # Adjust neuron indices for remaining neurons in the same layer
        for later_layer_idx in range(layer_idx + 1, len(self.layers)):
            for neuron in self.layers[later_layer_idx]:
                if 'connections' in neuron:
                    adjusted_connections = []
                    for src_layer, src_neuron, weight in neuron['connections']:
                        if src_layer == layer_idx and src_neuron > neuron_idx:
                            # Adjust index down by 1 since we removed a neuron
                            adjusted_connections.append((src_layer, src_neuron - 1, weight))
                        else:
                            adjusted_connections.append((src_layer, src_neuron, weight))
                    neuron['connections'] = adjusted_connections

        # Remove empty hidden layers to keep network clean
        if len(self.layers[layer_idx]) == 0:
            self.layers.pop(layer_idx)

            # Adjust all layer indices in connections that reference layers after the removed one
            for layer in self.layers:
                for neuron in layer:
                    if 'connections' in neuron:
                        adjusted_connections = []
                        for src_layer, src_neuron, weight in neuron['connections']:
                            if src_layer > layer_idx:
                                adjusted_connections.append((src_layer - 1, src_neuron, weight))
                            else:
                                adjusted_connections.append((src_layer, src_neuron, weight))
                        neuron['connections'] = adjusted_connections

    def _add_layer(self):
        """Add a new hidden layer at a random position with at least one neuron."""
        if len(self.layers) < 2:
            return  # Need at least input and output layers

        # Choose a position between input and output layers
        insert_idx = random.randint(1, len(self.layers) - 1)
        # Create a new hidden neuron
        new_neuron = {
            'type': 'hidden',
            'id': f'hidden_{random.randint(1000, 9999)}',
            'bias': random.uniform(-1, 1),
            'connections': []
        }
        # Connect to all neurons in the previous layer
        for prev_idx in range(len(self.layers[insert_idx - 1])):
            if random.random() < 0.5:
                new_neuron['connections'].append((insert_idx - 1, prev_idx, random.uniform(-2, 2)))
        # Insert the new layer
        self.layers.insert(insert_idx, [new_neuron])
        # Connect neurons in the next layer to the new neuron
        if insert_idx + 1 < len(self.layers):
            for neuron in self.layers[insert_idx + 1]:
                if 'connections' in neuron and random.random() < 0.5:
                    neuron['connections'].append((insert_idx, 0, random.uniform(-2, 2)))

    def _ensure_network_connectivity(self):
        """Ensure the network maintains basic connectivity from inputs to outputs."""
        # Check if output neurons have any connections
        output_layer = self.layers[-1]

        for i, output_neuron in enumerate(output_layer):
            if 'connections' not in output_neuron or len(output_neuron['connections']) == 0:
                # Output neuron has no connections - reconnect to input layer
                for j in range(self.input_size):
                    if not self._connection_exists(output_neuron, 0, j):
                        output_neuron['connections'].append((0, j, random.uniform(-2, 2)))
                        break  # Add at least one connection

        # Ensure at least one path exists from input to output
        if len(self.layers) > 2:  # Has hidden layers
            # Check if any hidden neurons are connected to inputs
            has_input_connection = False
            for layer_idx in range(1, len(self.layers) - 1):  # Hidden layers
                for neuron in self.layers[layer_idx]:
                    if 'connections' in neuron:
                        for src_layer, src_neuron, weight in neuron['connections']:
                            if src_layer == 0:  # Connected to input
                                has_input_connection = True
                                break
                    if has_input_connection:
                        break
                if has_input_connection:
                    break

            # If no hidden neuron connects to input, create one
            if not has_input_connection and len(self.layers) > 2:
                first_hidden_layer = self.layers[1]
                if len(first_hidden_layer) > 0:
                    first_neuron = first_hidden_layer[0]
                    if 'connections' in first_neuron:
                        # Add connection to first input
                        if not self._connection_exists(first_neuron, 0, 0):
                            first_neuron['connections'].append((0, 0, random.uniform(-2, 2)))

    def get_structure_info(self):
        """Return information about the network structure."""
        info = {
            'total_layers': len(self.layers),
            'layer_sizes': [len(layer) for layer in self.layers],
            'total_connections': 0,
            'total_neurons': sum(len(layer) for layer in self.layers),
            'network_cost': self.network_cost
        }

        for layer in self.layers[1:]:
            for neuron in layer:
                if 'connections' in neuron:
                    info['total_connections'] += len(neuron['connections'])

        return info

    def calculate_network_cost(self):
        """
        Estimate the computational cost of the network.
        Cost is defined as the total number of connections plus the number of neurons
        (i.e., total multiply-accumulate operations and activations per forward pass).
        """
        total_connections = 0
        total_neurons = 0
        for layer in self.layers[1:]:  # Skip input layer (no computation)
            for neuron in layer:
                total_neurons += 1
                if 'connections' in neuron:
                    total_connections += len(neuron['connections'])
        return total_connections + total_neurons


class CellBrain(BehavioralModel):
    """
    Enhanced CellBrain using a flexible neural network with input normalization.
    """

    def __init__(self, neural_network=None, input_ranges=None):
        super().__init__()

        # Define input and output keys
        self.input_keys = ['distance', 'angle', 'current_speed', 'current_angular_velocity']
        self.output_keys = ['linear_acceleration', 'angular_acceleration']

        # Initialize inputs and outputs
        self.inputs = {key: 0.0 for key in self.input_keys}
        self.outputs = {key: 0.0 for key in self.output_keys}

        # Set input ranges for normalization
        default_ranges = {
            'distance': (0, 50),
            'angle': (-180, 180),
            'current_speed': (-MAX_VELOCITY, MAX_VELOCITY),
            'current_angular_velocity': (-MAX_ROTATIONAL_VELOCITY, MAX_ROTATIONAL_VELOCITY)
        }
        self.input_ranges = input_ranges if input_ranges is not None else default_ranges

        # Use provided network or create new one
        if neural_network is None:
            self.neural_network = FlexibleNeuralNetwork(
                input_size=len(self.input_keys),
                output_size=len(self.output_keys)
            )
        else:
            self.neural_network = neural_network

    def _normalize_input(self, key, value):
        min_val, max_val = self.input_ranges.get(key, (0.0, 1.0))
        # Avoid division by zero
        if max_val == min_val:
            return 0.0
        # Normalize to [-1, 1]
        return 2 * (value - min_val) / (max_val - min_val) - 1

    def tick(self, input_data) -> dict:
        """
        Process inputs through neural network and produce outputs.

        :param input_data: Dictionary containing input values
        :return: Dictionary with output values
        """
        # Update internal input state
        for key in self.input_keys:
            self.inputs[key] = input_data.get(key, 0.0)

        # Normalize inputs
        input_array = [self._normalize_input(key, self.inputs[key]) for key in self.input_keys]

        # Process through neural network
        output_array = self.neural_network.forward(input_array)

        # Map outputs back to dictionary
        self.outputs = {
            key: output_array[i] if i < len(output_array) else 0.0
            for i, key in enumerate(self.output_keys)
        }

        return self.outputs.copy()

    def mutate(self, mutation_rate=0.1):
        """
        Create a mutated copy of this CellBrain.

        :param mutation_rate: Rate of mutation for the neural network
        :return: New CellBrain with mutated neural network
        """
        mutated_network = self.neural_network.mutate(mutation_rate)
        return CellBrain(neural_network=mutated_network, input_ranges=self.input_ranges.copy())

    def get_network_info(self):
        """Get information about the underlying neural network."""
        return self.neural_network.get_structure_info()

    def __repr__(self):
        inputs = {key: round(value, 5) for key, value in self.inputs.items()}
        outputs = {key: round(value, 5) for key, value in self.outputs.items()}
        network_info = self.get_network_info()

        return (f"CellBrain(inputs={inputs}, outputs={outputs}, "
                f"network_layers={network_info['layer_sizes']}, "
                f"connections={network_info['total_connections']})")