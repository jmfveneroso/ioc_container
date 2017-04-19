#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#include <vector>
#include <iostream>
#include <sstream>
#include <math.h>

namespace NeuralNetwork {

struct TrainingCase {
  std::vector<double> inputs;
  std::vector<double> results;
  TrainingCase(std::vector<double> inputs, std::vector<double> results) 
    : inputs(inputs), results(results) {
  }
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

  double ActivationFunction(const double& x) {
    return (double) 1 / (1 + exp(-x));
  }

  std::string ToString() {
    std::stringstream ss;
    ss << "    Neuron " << id << std::endl
       << "      bias: " << bias << std::endl
       << "      weights: " << std::endl;
    for (size_t i = 0; i < weight_derivatives.size(); ++i) {
      ss << "        " << i << " value: " << weights[i];
      ss << ", derivative: " << weight_derivatives[i] << std::endl;
    }
    ss << "      result: " << result << std::endl;
    return ss.str();
  }

  void Clear() {
    result = 0;
    for (size_t i = 0; i < weight_derivatives.size(); ++i)
      weight_derivatives[i] = 0;
  }

  double Calculate(std::vector<double> inputs) {
    if (inputs.size() != weights.size()) throw new std::runtime_error("Invalid input vector.");

    result = bias;
    for (size_t i = 0; i < inputs.size(); ++i) result += weights[i] * inputs[i];
    result = ActivationFunction(result);
    return result;
  }
};

using Layer = std::vector<Neuron>;

class NeuralNet {
  std::vector<Layer> hidden_layers_;
  Layer output_layer_;
  double learning_rate_ = 0.25;
  double momentum_ = 0.0001;

 public:
  NeuralNet();
  // void LoadConfigurationFromFile(const char[]);
  // void SaveConfigurationToFile(const char[]);

  void set_learning_rate(double learning_rate) { learning_rate_ = learning_rate; }
  void SetOutputLayer(Layer);
  void AddHiddenLayer(Layer);
  void Train(const TrainingCase&);
  std::vector<double> Predict(std::vector<double>);
  void UpdateWeights();
  std::string ToString();
};

} // End of namespace.

#endif
