from abc import ABC, abstractmethod
from collections import defaultdict
from types import NotImplementedType

import numpy as np
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit import Parameter
from qiskit.result import marginal_counts
from qiskit_aer import AerSimulator
from sklearn.metrics import log_loss


def output_mapping(binary_counts, n_classes):
    """
    Maps binary count output to n classes
    """
    class_preds = defaultdict(int, {i: 0 for i in range(n_classes)})
    for binary, count in binary_counts.items():
        c = int(binary, base=2) % n_classes
        class_preds[c] += count

    return class_preds


BACKEND = AerSimulator()


# Circuits
def Custom_UnitaryGate1(parameters):
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.rz(parameters[0], 1)
    qc.ry(parameters[1], 0)
    qc.cx(0, 1)
    return qc.to_gate(label="U1")


def Custom_UnitaryGate2(parameters):
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.rx(parameters[0], 1)
    qc.rx(parameters[1], 0)
    qc.rz(parameters[2], 0)
    qc.rz(parameters[3], 1)
    qc.cx(0, 1)
    return qc.to_gate(label="U2")


def Custom_UnitaryGateV(parameters):
    qc = QuantumCircuit(1)
    qc.rx(parameters[0], 0)
    qc.ry(parameters[1], 0)
    qc.rz(parameters[2], 0)
    return qc.to_gate(label="V")


def circuit1(features, parameters=None):
    qc_1 = QuantumCircuit(4, 4)
    for i in range(4):
        qc_1.rx(qubit=i, theta=features[i])
    qc_1.barrier()

    # qc_1_unitary_V2 = UnitaryGate([[1,0], [0,1]])
    params = [Parameter(f"{i}") for i in range(9)]
    qc_1.append(Custom_UnitaryGate1([params[0], params[1]]), [3, 2])
    qc_1.append(Custom_UnitaryGate1([params[0], params[1]]), [1, 0])
    qc_1.barrier()

    qc_1.measure(qubit=2, cbit=2)
    qc_1.append(
        Custom_UnitaryGateV([params[2], params[3], params[4]]).control(1), [2, 3]
    )

    qc_1.measure(qubit=0, cbit=0)
    qc_1.append(
        Custom_UnitaryGateV([params[2], params[3], params[4]]).control(1), [0, 1]
    )

    qc_1.barrier()
    qc_1.append(
        Custom_UnitaryGate2([params[5], params[6], params[7], params[8]]), [3, 1]
    )
    qc_1.measure(qubit=1, cbit=1)
    qc_1.measure(qubit=3, cbit=3)

    if parameters is not None:
        qc_1 = qc_1.assign_parameters(parameters)
    return qc_1


def circuit2(features, trainable_parameters=None, layers=2):
    """
    Circuit implementing real amplitudes
    """
    input_size = len(features)
    qc = QuantumCircuit(input_size)
    for i in range(input_size):
        qc.rx(features[i], i)
    qc.barrier()

    for i in range(layers):
        for j in range(input_size):
            qc.ry(Parameter(f"phi{i}{j}"), j)
        for j in range(input_size - 1, 0, -1):
            qc.cx(j, j - 1)
        qc.barrier()

    if trainable_parameters is not None:
        qc = qc.assign_parameters(trainable_parameters)
    qc.measure_all()
    return qc


def circuit3(features, trainable_parameters=None, layers=2):
    input_size = len(features)
    qc = QuantumCircuit(input_size)

    # Step 1: Feature Encoding
    for i in range(input_size):
        # Legger til en Hadamard-port på hver qubit, som gir en superposisjonstilstand.
        qc.h(i)
        # RZ-porten er en av flere mulige porter vi kan bruke for encoding.
        # Merk, kan bruke (RX, RY eller RZ).
        qc.rz(features[i], i)
    qc.barrier()

    # Step 2: Variational Layers
    for i in range(layers):
        for j in range(input_size):
            qc.rx(Parameter(f"theta{i}{j}"), j)  # Justerbar RX-rotasjon på hver qubit.
            qc.ry(Parameter(f"phi{i}{j}"), j)  # Legger til en justerbar RY-rotasjon.

        # For å skape entanglement, legger vi til CX-porter mellom hvert par av qubits.
        for j in range(
            0, input_size - 1, 2
        ):  # For å skape entanglement, legger vi til CX-porter mellom hvert par av qubits.
            qc.cx(j, j + 1)
        qc.cx(1, 2)
        qc.barrier()

    # Step 3: Parameter Binding and Measurement
    if trainable_parameters is not None:
        qc = qc.assign_parameters(trainable_parameters)
    qc.measure_all()

    return qc


class BaseModel(ABC):
    """
    Base model for QNNs
    """

    @abstractmethod
    def __init__(
        self,
        learning_rate=0.01,
        prediction_shots=1000,
        gradient_shots=100,
        epsilon=1,
        patience=3,
        min_delta=0.2,
        seed=None,
        **kwargs,
    ):
        self.learning_rate = learning_rate
        self.gradient_shots = gradient_shots
        self.prediction_shots = prediction_shots
        self.train_loss = []
        self.val_loss = []
        self.epsilon = epsilon
        self.seed = seed

        self.pass_manager = generate_preset_pass_manager(
            backend=BACKEND, optimization_level=1
        )

        self.circuit_params = []
        self.parameters: np.ndarray = NotImplementedType
        self.circuit_func = NotImplementedType
        np.random.seed(seed)

    def measure_circuit(
        self, qc: QuantumCircuit, shots: int = 1000, qbits: list | None = None
    ):
        isa_circuit = self.pass_manager.run(qc)
        result = BACKEND.run(isa_circuit, shots=shots).result()
        if qbits is not None:
            return marginal_counts(result, qbits).get_counts()
        else:
            return result.get_counts()

    def gradient(self, data, targets):
        size = len(self.parameters)
        params_plus = np.broadcast_to(
            self.parameters, (size, size)
        ) + self.epsilon * np.eye(size)
        params_minus = np.broadcast_to(
            self.parameters, (size, size)
        ) - self.epsilon * np.eye(size)
        losses_plus = np.array(
            [
                self._loss(data, targets, self.gradient_shots, params)
                for params in params_plus
            ]
        )
        losses_minus = np.array(
            [
                self._loss(data, targets, self.gradient_shots, params)
                for params in params_minus
            ]
        )
        return np.sum((losses_plus - losses_minus) / (2 * self.epsilon)) / size

    def _loss(self, data, targets, shots, parameters=None):
        if parameters is None:
            parameters = self.parameters
        preds = np.array(
            [
                self._predict(x, shots=shots, parameters=parameters, probabilities=True)
                for x in data
            ]
        )
        return log_loss(targets, preds, labels=[0, 1, 2])

    def fit(
        self,
        epochs,
        data,
        targets,
        val_data=None,
        val_targets=None,
        patience=1,
        min_delta=0.01,
        max_delta=0.2,
    ):
        best_val_loss = float("inf")
        patience_counter = 0

        for i in range(epochs):
            if patience_counter == patience:
                print("Early stopping triggered.")
                break

            print(f"Epoch {i+1}", end=" ")
            gradient = self.gradient(data, targets)
            self.parameters -= self.learning_rate * gradient

            train_loss = self._loss(data, targets, shots=self.prediction_shots)
            print(f"Train loss: {train_loss}", end=" ")
            self.train_loss.append(train_loss)

            if val_data is not None and val_targets is not None:
                val_loss = self._loss(
                    val_data, val_targets, shots=self.prediction_shots
                )
                self.val_loss.append(val_loss)
                print(f"Validation loss: {val_loss}")

                # Check for improvement
                if val_loss >= best_val_loss + max_delta:
                    print("Val loss exceeding max delta. Stopping early!")
                    break
                elif val_loss >= best_val_loss + min_delta:
                    patience_counter += 1
                    print(f"Worse! Patience is increased to {patience_counter}")
                else:
                    best_val_loss = val_loss
                    patience_counter = 0

        return self

    def _predict(self, x, shots=1000, parameters=None, probabilities=False):
        if parameters is None:
            parameters = self.parameters

        circuit = self.circuit_func(x, parameters, *self.circuit_params)
        results = self.measure_circuit(circuit, shots)
        class_output = output_mapping(results, 3)
        if probabilities:
            class_output = sorted(output_mapping(results, 3).items())
            prediction = np.array([n / shots for _, n in class_output])
            return prediction
        else:
            prediction = max(class_output, key=lambda x: class_output[x])
            return prediction

    def predict(self, X):
        return np.array([self._predict(x, shots=self.prediction_shots) for x in X])

    def predict_proba(self, X):
        return np.array(
            [
                self._predict(x, shots=self.prediction_shots, probabilities=True)
                for x in X
            ]
        )


class Model1(BaseModel):
    """
    Model for convolutional neral network
    """

    def __init__(
        self,
        learning_rate=0.01,
        prediction_shots=1000,
        gradient_shots=100,
        epsilon=1,
        seed=None,
        **kwargs,
    ):
        super().__init__(
            learning_rate, prediction_shots, gradient_shots, epsilon, seed, **kwargs
        )
        self.parameters = np.random.uniform(low=0, high=2 * np.pi, size=9)
        self.circuit_func = circuit1


class Model2(BaseModel):
    """
    Model for real amplitudes
    """

    def __init__(
        self,
        layers=2,
        learning_rate=0.01,
        prediction_shots=1000,
        gradient_shots=100,
        epsilon=1,
        seed=None,
        **kwargs,
    ):
        super().__init__(
            learning_rate, prediction_shots, gradient_shots, epsilon, seed, **kwargs
        )
        self.parameters = np.random.uniform(low=0, high=2 * np.pi, size=(layers * 4,))
        self.layers = layers
        self.circuit_func = circuit2
        self.circuit_params = [layers]


class Model3(BaseModel):
    """
    Model for real amplitudes
    """

    def __init__(
        self,
        layers=4,
        learning_rate=0.01,
        prediction_shots=1000,
        gradient_shots=100,
        epsilon=1,
        seed=None,
        **kwargs,
    ):
        super().__init__(
            learning_rate, prediction_shots, gradient_shots, epsilon, seed, **kwargs
        )
        self.parameters = np.random.uniform(
            low=0, high=2 * np.pi, size=(2 * layers * 4,)
        )
        self.layers = layers
        self.circuit_func = circuit3
        self.circuit_params = [layers]
