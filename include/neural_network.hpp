#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#include <vector>
#include <iostream>

namespace NeuralNetwork {

struct Neuron {
  std::vector<double> weights;
  std::vector<double> weight_derivatives;
  double result;
  Neuron() : result(0) {}
  Neuron(std::vector<double> weights) : weights(weights), result(0) {
    weight_derivatives = std::vector<double>(weights.size(), 0);
  }

  void Clear() {
    result = 0;
    for (size_t i = 0; i < weight_derivatives.size(); ++i) {
      weight_derivatives[i] = 0;
    }
  }
};

using Layer = std::vector<Neuron>;

struct Configuration {
  double threshold;
  Layer input_layer;
  Layer output_layer;
  std::vector<Layer> hidden_layers;
};

class INeuralNetwork {
 public:
  virtual void LoadConfiguration(Configuration&) = 0;
  virtual double Calculate(std::vector<double>) = 0;
  virtual Layer& GetHiddenLayer(size_t i = 0) = 0;
  virtual Layer& GetOutputLayer() = 0;
  virtual Layer& GetInputLayer() = 0;
  virtual size_t GetNumHiddenLayers() = 0;
  // virtual void LoadFromFile(const char[]) = 0;
};

class NeuralNet : public INeuralNetwork {
  Configuration cfg_;

  void Init();
  void FeedForward(Layer&, Layer&, bool);
  double ActivationFunction(const double&);

 public:
  NeuralNet();
  void LoadConfiguration(Configuration&);
  // void LoadConfigurationFromFile(const char[]);
  // void SaveConfigurationToFile(const char[]);
  double Calculate(std::vector<double>);
  
  size_t GetNumHiddenLayers() { return cfg_.hidden_layers.size(); }
  Layer& GetInputLayer() { return cfg_.input_layer; }
  Layer& GetOutputLayer() { return cfg_.output_layer; }
  Layer& GetHiddenLayer(size_t i = 0) { return cfg_.hidden_layers[i]; }
};

}; // End of namespace.

#endif
