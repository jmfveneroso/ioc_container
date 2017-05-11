#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#include <vector>
#include <string>

namespace NeuralNetwork {

struct TrainingCase {
  std::vector<double> inputs;
  std::vector<double> results;

  TrainingCase() {}
  TrainingCase(std::vector<double> inputs, std::vector<double> results) 
    : inputs(inputs), results(results) {}
};

struct Neuron { 
  size_t id;
  double bias;
  double bias_derivative;
  double prev_bias_delta;
  std::vector<double> weights;
  std::vector<double> weight_derivatives;
  std::vector<double> prev_deltas;
  double result;
  double delta;

  Neuron() : result(0), delta(0) {
    static size_t id_counter = 0;
    id = ++id_counter;
  }

  Neuron(double bias, std::vector<double> weights) : Neuron() {
    this->bias = bias;
    this->weights = weights;
    weight_derivatives = std::vector<double>(weights.size(), 0);
    prev_deltas = std::vector<double>(weights.size(), 0);
  }
};

using Layer = std::vector<Neuron>;

class NeuralNet {
  std::vector<Layer> hidden_layers_;
  Layer output_layer_;
  double learning_rate_ = 0.3;
  double momentum_ = 0.9;
  size_t num_training_cases_ = 0;

  double GetOutputDerivative(Neuron&, double);
  void UpdateNeuron(Neuron&, bool);
  double ActivationFunction(const double&);
  double CalculateNeuron(Neuron&, const std::vector<double>&);
  double CalculateNeuron(Neuron&, const Layer&);
  void ClearDerivatives();
  std::string NeuronToString(Neuron&);

 public:
  NeuralNet();
  void set_learning_rate(double learning_rate) { learning_rate_ = learning_rate; }
  void LoadFromFile(const std::string&);
  void SaveToFile(const std::string&);
  void SetOutputLayer(Layer);
  void AddHiddenLayer(Layer);
  std::vector<double> Predict(const std::vector<double>&);
  void Train(const TrainingCase&);
  void UpdateWeights();
  std::string ToString();
};

} // End of namespace.

#endif
