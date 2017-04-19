#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#include <vector>

namespace NeuralNetwork {

struct TrainingCase {
  std::vector<double> inputs;
  std::vector<double> results;

  TrainingCase(std::vector<double> inputs, std::vector<double> results) 
    : inputs(inputs), results(results) {}
};

struct Neuron { 
  size_t id;
  double bias;
  std::vector<double> weights;
  std::vector<double> weight_derivatives;
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
  }
};

using Layer = std::vector<Neuron>;

class NeuralNet {
  std::vector<Layer> hidden_layers_;
  Layer output_layer_;
  double learning_rate_ = 0.25;
  double momentum_ = 0.0001;

  void UpdateNeuron(Neuron&);
  double ActivationFunction(const double&);
  double CalculateNeuron(Neuron&, std::vector<double>);
  std::string NeuronToString(Neuron&);

 public:
  NeuralNet();
  void set_learning_rate(double learning_rate) { learning_rate_ = learning_rate; }
  void LoadFromFile(const std::string&);
  void SaveToFile(const std::string&);
  void SetOutputLayer(Layer);
  void AddHiddenLayer(Layer);
  std::vector<double> Predict(std::vector<double>);
  void Train(const TrainingCase&);
  void UpdateWeights();
  std::string ToString();
};

} // End of namespace.

#endif
