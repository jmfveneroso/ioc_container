#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#include <vector>
#include "activation_function.hpp"

namespace NeuralNetwork {

struct Neuron {
  std::vector<double> weights;
  double result;
  Neuron() : result(0) {}
  Neuron(std::vector<double>& weights) : weights(weights), result(0) {}
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
  virtual int Calculate(std::vector<double>) = 0;
  virtual Layer& GetHiddenLayer(size_t i = 0) = 0;
  virtual Layer& GetOutputLayer() = 0;
  // virtual void LoadFromFile(const char[]) = 0;
};

class NeuralNet : public INeuralNetwork {
  std::shared_ptr<IActivationFunction> activation_fn_;
  Configuration cfg_;

  void Init();
  void FeedForward(Layer&, Layer&);

 public:
  NeuralNet(std::shared_ptr<IActivationFunction>);
  void LoadConfiguration(Configuration&);
  // void LoadConfigurationFromFile(const char[]);
  // void SaveConfigurationToFile(const char[]);
  int Calculate(std::vector<double>);
  
  Layer& GetHiddenLayer(size_t i = 0) { return cfg_.hidden_layers[i]; }
  Layer& GetOutputLayer() { return cfg_.output_layer; }
};

}; // End of namespace.

#endif
