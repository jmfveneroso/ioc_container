#include "neural_network.hpp"
#include <iostream>
#include <math.h>

namespace NeuralNetwork {

NeuralNet::NeuralNet() {}

void NeuralNet::SetOutputLayer(Layer output_layer) {
  output_layer_ = output_layer;
}

void NeuralNet::AddHiddenLayer(Layer hidden_layer) {
  hidden_layers_.push_back(hidden_layer);
}

std::vector<double> NeuralNet::Predict(std::vector<double> inputs) {
  // Feed the hidden layers.
  std::vector<double> results;
  for (size_t i = 0; i < hidden_layers_.size(); ++i) {
    results = std::vector<double>(hidden_layers_[i].size());
    for (size_t j = 0; j < hidden_layers_[i].size(); ++j)
      results[j] = hidden_layers_[i][j].Calculate(inputs);
    inputs = results; 
  }

  // Feed the output layer.
  results = std::vector<double>(output_layer_.size());
  for (size_t i = 0; i < output_layer_.size(); ++i)
    results[i] = output_layer_[i].Calculate(inputs);
     
  return results;
}

void NeuralNet::Train(const TrainingCase& training_case) {
  Predict(training_case.inputs);

  for (size_t i = 0; i < output_layer_.size(); ++i) {
    Neuron& neuron = output_layer_[i];
    double error = training_case.results[i] - neuron.result;
    neuron.delta = neuron.result * (1 - neuron.result) * error;
  }

  Layer* prev_layer = &output_layer_;
  for (int i = hidden_layers_.size() - 1; i >= 0; --i) {
    Layer& hidden_layer = hidden_layers_[i];
    for (size_t j = 0; j < hidden_layer.size(); ++j) {
      Neuron& neuron = hidden_layer[j];
      double total_output_derivative = 0;
      for (size_t k = 0; k < prev_layer->size(); ++k) {
        Neuron& output_neuron = (*prev_layer)[k];
        total_output_derivative += output_neuron.delta * output_neuron.weights[j];
        output_neuron.weight_derivatives[j] = output_neuron.delta * neuron.result;
      }
      neuron.delta = neuron.result * (1 - neuron.result) * total_output_derivative;
    }
    prev_layer = &hidden_layer;
  }

  for (size_t i = 0; i < prev_layer->size(); ++i) {
    Neuron& neuron = (*prev_layer)[i];
    for (size_t j = 0; j < training_case.inputs.size(); ++j) {
      neuron.weight_derivatives[j] = neuron.delta * training_case.inputs[j];
    }
  }
}

void NeuralNet::UpdateWeights() {
  for (size_t i = 0; i < hidden_layers_.size(); ++i) {
    Layer& hidden_layer = hidden_layers_[i];
    for (size_t j = 0; j < hidden_layer.size(); ++j) {
      Neuron& neuron = hidden_layer[j];
      neuron.bias += learning_rate_ * neuron.delta + momentum_ * neuron.bias;
      for (size_t k = 0; k < neuron.weights.size(); ++k) {
        neuron.weights[k] += learning_rate_ * neuron.weight_derivatives[k] + momentum_ * neuron.weights[k];
      }
    }
  }

  for (size_t i = 0; i < output_layer_.size(); ++i) {
    Neuron& neuron = output_layer_[i];
    neuron.bias += learning_rate_ * neuron.delta + momentum_ * neuron.bias;
    for (size_t j = 0; j < neuron.weights.size(); ++j) {
      neuron.weights[j] += learning_rate_ * neuron.weight_derivatives[j] + momentum_ * neuron.weights[j];
    }
  }
}

std::string NeuralNet::ToString() {
  std::stringstream ss;
  ss << "Neural Network" << std::endl;
  for (size_t i = 0; i < hidden_layers_.size(); ++i) {
    ss << "  Hidden Layer [" << i << "]:" << std::endl;
    for (size_t j = 0; j < hidden_layers_[i].size(); ++j) {
      ss << hidden_layers_[i][j].ToString() << std::endl;
    }
  }

  ss << "  Output Layer (" << output_layer_.size() << "):" << std::endl;
  for (size_t i = 0; i < output_layer_.size(); ++i) {
    ss << output_layer_[i].ToString() << std::endl;
  }
  return ss.str();
}

} // End of namespace.
