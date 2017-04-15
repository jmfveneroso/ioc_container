#include "neural_network.hpp"
#include <iostream>
#include <math.h>

namespace NeuralNetwork {

NeuralNet::NeuralNet() {}

void NeuralNet::Init() {
  size_t last_layer_size = cfg_.input_layer.size() - 1;
  for (size_t i = 0; i < cfg_.hidden_layers.size(); ++i) {
    size_t layer_size = cfg_.hidden_layers[i].size();
    for (size_t j = 0; j < layer_size; ++j)
      cfg_.hidden_layers[i][j].weights.resize(last_layer_size + 1);

    last_layer_size = layer_size;
  }

  for (size_t i = 0; i < cfg_.output_layer.size(); ++i)
    cfg_.output_layer[i].weights.resize(last_layer_size + 1);
}

double NeuralNet::ActivationFunction(const double& x) {
  return 1 / (1 + exp(-x));
}

void NeuralNet::FeedForward(Layer& current, Layer& next, bool is_input) {
  for (size_t i = 0; i < next.size(); ++i) {
    next[i].result = next[i].weights[0] * 1; // Bias.
    for (size_t j = 0; j < current.size(); ++j) {
      if (is_input) {
        next[i].result += next[i].weights[j + 1] * current[j + 1].result;
      } else {  
        next[i].result += next[i].weights[j + 1] * current[j].result;
      }
    }
    next[i].result = ActivationFunction(next[i].result);
  }
}

void NeuralNet::LoadConfiguration(Configuration& config) {
  cfg_ = config;
  Init();
}

double NeuralNet::Calculate(std::vector<double> inputs) {
  if (inputs.size() != cfg_.input_layer.size())
    throw new std::runtime_error("Input vector has the wrong size.");

  // Feed the input layer.
  for (size_t i = 0; i < inputs.size(); ++i)
    cfg_.input_layer[i].result = inputs[i]; 

  // Start feeding the hidden layers.
  bool is_input = true;
  Layer current_layer = cfg_.input_layer;
  for (size_t i = 0; i < cfg_.hidden_layers.size(); ++i) {
    FeedForward(current_layer, cfg_.hidden_layers[i], is_input);
    is_input = false;
    current_layer = cfg_.hidden_layers[i];
  }

  // Feed the output layer.
  FeedForward(current_layer, cfg_.output_layer, is_input);

  if (cfg_.output_layer.size() == 1) {
    return cfg_.output_layer[0].result;
  }

  return 0;
}

}; // End of namespace.
