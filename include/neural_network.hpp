#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#include <vector>
#include <iostream>
#include <sstream>

namespace NeuralNetwork {

struct Neuron { 
  size_t id;
  std::vector<double> weights;
  std::vector<double> weight_derivatives;
  double result;

  Neuron() : result(0) {
    static size_t id_counter = 0;
    id = ++id_counter;
  }

  Neuron(std::vector<double> weights) : Neuron() {
    this->weights = weights;
    weight_derivatives = std::vector<double>(weights.size(), 0);
  }

  std::string ToString() {
    std::stringstream ss;
    ss << "    Neuron " << id << std::endl
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
  virtual std::string ToString() = 0;
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

  std::string ToString() {
    std::stringstream ss;
    ss << "Neural Network" << std::endl;
    ss << "  Input Layer (" << cfg_.input_layer.size() << "):" << std::endl;
    for (size_t i = 0; i < cfg_.input_layer.size(); ++i) {
      ss << cfg_.input_layer[i].ToString() << std::endl;
    }

    for (size_t i = 0; i < cfg_.hidden_layers.size(); ++i) {
      ss << "  Hidden Layer [" << i << "]:" << std::endl;
      for (size_t j = 0; j < cfg_.hidden_layers[i].size(); ++j) {
        ss << cfg_.hidden_layers[i][j].ToString() << std::endl;
      }
    }

    ss << "  Output Layer (" << cfg_.output_layer.size() << "):" << std::endl;
    for (size_t i = 0; i < cfg_.output_layer.size(); ++i) {
      ss << cfg_.output_layer[i].ToString() << std::endl;
    }
    return ss.str();
  }
};

} // End of namespace.

#endif
