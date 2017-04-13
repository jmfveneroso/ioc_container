#include "neural_network.hpp"
#include <iostream>

namespace NeuralNetwork {

NeuralNet::NeuralNet(std::shared_ptr<IActivationFunction> activation_fn) 
  : activation_fn_(activation_fn) {
}

void NeuralNet::Init() {
  size_t last_layer_size = cfg_.input_layer.size();
  for (size_t i = 0; i < cfg_.hidden_layers.size(); ++i) {
    size_t layer_size = cfg_.hidden_layers[i].size();
    for (size_t j = 0; j < layer_size; ++j)
      cfg_.hidden_layers[i][j].weights.resize(last_layer_size);

    last_layer_size = layer_size;
  }

  for (size_t i = 0; i < cfg_.output_layer.size(); ++i)
    cfg_.output_layer[i].weights.resize(last_layer_size);
}

void NeuralNet::FeedForward(Layer& current, Layer& next) {
  for (size_t i = 0; i < next.size(); ++i) {
    next[i].result = 0;
    for (size_t j = 0; j < current.size(); ++j) {
      next[i].result += next[i].weights[j] * current[j].result;
    }
    next[i].result = activation_fn_->GetResult(next[i].result);
  }
}

void NeuralNet::LoadConfiguration(Configuration& config) {
  cfg_ = config;
  Init();
}

int NeuralNet::Calculate(std::vector<double> inputs) {
  if (inputs.size() != cfg_.input_layer.size())
    throw new std::runtime_error("Input vector has the wrong size.");

  // Feed the input layer.
  for (size_t i = 0; i < inputs.size(); ++i)
    cfg_.input_layer[i].result = inputs[i]; 

  // Start feeding the hidden layers.
  Layer& current_layer = cfg_.input_layer;
  for (size_t i = 0; i < cfg_.hidden_layers.size(); ++i) {
    FeedForward(current_layer, cfg_.hidden_layers[i]);
    current_layer = cfg_.hidden_layers[i];
  }

  // Feed the output layer.
  FeedForward(current_layer, cfg_.output_layer);

  if (cfg_.output_layer.size() == 1) {
    return (cfg_.output_layer[0].result > cfg_.threshold) ? 1 : 0;
  }

  return 0;
}

}; // End of namespace.
