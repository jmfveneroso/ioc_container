#include "neural_network.hpp"
#include <iostream>
#include <math.h>
#include <fstream>
#include <sstream>

namespace NeuralNetwork {

NeuralNet::NeuralNet() {}

void NeuralNet::SetOutputLayer(Layer output_layer) {
  output_layer_ = output_layer;
}

void NeuralNet::AddHiddenLayer(Layer hidden_layer) {
  hidden_layers_.push_back(hidden_layer);
}

double NeuralNet::ActivationFunction(const double& x) {
  // Sigmoid.
  return (double) 1 / (1 + exp(-x));
}

double NeuralNet::CalculateNeuron(Neuron& neuron, const std::vector<double>& inputs) {
  if (inputs.size() != neuron.weights.size()) throw new std::runtime_error("Invalid input size.");

  neuron.result = neuron.bias;
  std::vector<double>::const_iterator it_1 = neuron.weights.begin(), it_2 = inputs.begin();
  for (size_t i = 0; i < inputs.size(); ++i) 
    neuron.result += neuron.weights[i] * inputs[i];

  neuron.result = ActivationFunction(neuron.result);
  if (neuron.result > 0.999) neuron.result -= 0.001;
  if (neuron.result < 0.001) neuron.result += 0.001;
  return neuron.result;
}

double NeuralNet::CalculateNeuron(Neuron& neuron, const Layer& prev_layer) {
  if (prev_layer.size() != neuron.weights.size()) throw new std::runtime_error("Invalid input size.");

  neuron.result = neuron.bias;
  for (size_t i = 0; i < prev_layer.size(); ++i) 
    neuron.result += neuron.weights[i] * prev_layer[i].result;

  neuron.result = ActivationFunction(neuron.result);
  if (neuron.result > 0.999) neuron.result -= 0.001;
  if (neuron.result < 0.001) neuron.result += 0.001;
  return neuron.result;
}

std::vector<double> NeuralNet::Predict(const std::vector<double>& inputs) {
  Layer* prev_layer = nullptr;
  
  // Feed the hidden layers.
  for (size_t i = 0; i < hidden_layers_.size(); ++i) {
    for (size_t j = 0; j < hidden_layers_[i].size(); ++j) {
      if (prev_layer == nullptr)
        CalculateNeuron(hidden_layers_[i][j], inputs);
      else 
        CalculateNeuron(hidden_layers_[i][j], *prev_layer);
    }
    prev_layer = &hidden_layers_[i];
  }

  // Feed the output layer.
  std::vector<double> results = std::vector<double>(output_layer_.size());
  for (size_t i = 0; i < output_layer_.size(); ++i)
    results[i] = CalculateNeuron(output_layer_[i], *prev_layer);
     
  return results;
}

void NeuralNet::ClearDerivatives() {
  for (size_t i = 0; i < output_layer_.size(); ++i) {
    output_layer_[i].bias_derivative = 0;
    output_layer_[i].prev_bias_delta = 0;
    std::fill(output_layer_[i].weight_derivatives.begin(), output_layer_[i].weight_derivatives.end(), 0);
    std::fill(output_layer_[i].prev_deltas.begin(), output_layer_[i].prev_deltas.end(), 0);
  }
  for (int i = hidden_layers_.size() - 1; i >= 0; --i) {
    for (size_t j = 0; j < hidden_layers_[i].size(); ++j) {
      hidden_layers_[i][j].bias_derivative = 0;
      hidden_layers_[i][j].prev_bias_delta = 0;
      std::fill(hidden_layers_[i][j].weight_derivatives.begin(), hidden_layers_[i][j].weight_derivatives.end(), 0);
      std::fill(hidden_layers_[i][j].prev_deltas.begin(), hidden_layers_[i][j].prev_deltas.end(), 0);
    }
  }
  num_training_cases_ = 0;
}

double NeuralNet::GetOutputDerivative(Neuron& neuron, double expected_result) {
  // Loss function: Mean squared error.
  // double error = expected_result - neuron.result;
  // return neuron.result * (1 - neuron.result) * error;

  // Loss function: Cross Entropy.
  // We are computing the inverse of the gradient, so the parameters were inverted
  // here. The derivative of the cross entropy function is: neuron.result - expected_result.
  return expected_result - neuron.result;
}

void NeuralNet::Train(const TrainingCase& training_case) {
  ++num_training_cases_;
  for (size_t i = 0; i < output_layer_.size(); ++i) {
    Neuron& neuron = output_layer_[i];

    neuron.delta = GetOutputDerivative(neuron, training_case.results[i]);
    neuron.bias_derivative += neuron.delta;
  }

  Layer* prev_layer = &output_layer_;
  for (int i = hidden_layers_.size() - 1; i >= 0; --i) {
    for (size_t j = 0; j < hidden_layers_[i].size(); ++j) {
      Neuron& neuron = hidden_layers_[i][j];

      // The output derivative holds the change provoked by the output of this
      // neuron in every other neuron at the next layer.
      double output_derivative = 0;
      for (size_t k = 0; k < prev_layer->size(); ++k) {
        Neuron& prev_neuron = (*prev_layer)[k];
        output_derivative += prev_neuron.delta * prev_neuron.weights[j];
        prev_neuron.weight_derivatives[j] += prev_neuron.delta * neuron.result;
      }
      neuron.delta = neuron.result * (1 - neuron.result) * output_derivative;
      neuron.bias_derivative += neuron.delta;
    }
    prev_layer = &hidden_layers_[i];
  }

  for (size_t i = 0; i < prev_layer->size(); ++i) {
    Neuron& neuron = (*prev_layer)[i];
    for (size_t j = 0; j < training_case.inputs.size(); ++j) {
      neuron.weight_derivatives[j] += neuron.delta * training_case.inputs[j];
    }
  }
}

void NeuralNet::UpdateNeuron(Neuron& neuron) {
  // The division by num_training_cases is required by the cross entropy
  // loss derivative.
  double delta = learning_rate_ * neuron.bias_derivative 
                 + momentum_ * neuron.prev_bias_delta;
  neuron.bias += delta / num_training_cases_;
  neuron.prev_bias_delta = delta / num_training_cases_;

  for (size_t k = 0; k < neuron.weights.size(); ++k) {
    double delta = learning_rate_ * neuron.weight_derivatives[k]
                   + momentum_ * neuron.prev_deltas[k];
    neuron.weights[k] += delta / num_training_cases_;
    neuron.prev_deltas[k] = delta / num_training_cases_;
  }
}

void NeuralNet::UpdateWeights() {
  for (size_t i = 0; i < hidden_layers_.size(); ++i)
    for (size_t j = 0; j < hidden_layers_[i].size(); ++j)
      UpdateNeuron(hidden_layers_[i][j]);

  for (size_t i = 0; i < output_layer_.size(); ++i)
    UpdateNeuron(output_layer_[i]);
  ClearDerivatives();
}

std::string NeuralNet::NeuronToString(Neuron& neuron) {
  std::stringstream ss;
  ss << "    Neuron " << neuron.id << std::endl
     << "      bias: " << neuron.bias << std::endl
     << "      result: " << neuron.result << std::endl
     << "      delta: " << neuron.delta << std::endl
     << "      weights: " << std::endl;
  for (size_t i = 0; i < neuron.weight_derivatives.size(); ++i) {
    ss << "        " << i << " value: " << neuron.weights[i];
    ss << ", derivative: " << neuron.weight_derivatives[i] << std::endl;
  }
  return ss.str();
}

std::string NeuralNet::ToString() {
  std::stringstream ss;
  ss << "Neural Network" << std::endl;
  for (size_t i = 0; i < hidden_layers_.size(); ++i) {
    ss << "  Hidden Layer [" << i << "]:" << std::endl;
    for (size_t j = 0; j < hidden_layers_[i].size(); ++j)
      ss << NeuronToString(hidden_layers_[i][j]) << std::endl;
  }

  ss << "  Output Layer (" << output_layer_.size() << "):" << std::endl;
  for (size_t i = 0; i < output_layer_.size(); ++i)
    ss << NeuronToString(output_layer_[i]) << std::endl;
  return ss.str();
}

void NeuralNet::LoadFromFile(const std::string& filename) {
  std::ifstream in_file(filename);
  if (!in_file.is_open()) return;

  size_t num_output_neurons = 0;
  in_file >> num_output_neurons;
  size_t num_weights = 0;
  in_file >> num_weights;

  output_layer_ = Layer(num_output_neurons);
  for (size_t i = 0; i < num_output_neurons; ++i) {
    Neuron neuron(0, std::vector<double>(num_weights));
    in_file >> neuron.bias;
    for (size_t j = 0; j < num_weights; ++j) in_file >> neuron.weights[j];
    output_layer_[i] = neuron;
  }
  
  size_t num_hidden_layers = 0;
  in_file >> num_hidden_layers;
  hidden_layers_ = std::vector<Layer>(num_hidden_layers);
  for (size_t i = 0; i < num_hidden_layers; ++i) {
    size_t num_hidden_neurons = 0;
    in_file >> num_hidden_neurons;
    in_file >> num_weights;

    for (size_t j = 0; j < num_hidden_neurons; ++j) {
      Neuron neuron(0, std::vector<double>(num_weights));
      in_file >> neuron.bias;
      for (size_t k = 0; k < num_weights; ++k) in_file >> neuron.weights[k];
      hidden_layers_[i].push_back(neuron);
    }
  }
  in_file.close();
}

void NeuralNet::SaveToFile(const std::string& filename) {
  std::ofstream out_file;
  out_file.open(filename);
  out_file << output_layer_.size() << std::endl;
  out_file << output_layer_[0].weights.size() << std::endl;
  for (size_t i = 0; i < output_layer_.size(); ++i) {
    out_file << output_layer_[i].bias << std::endl;
    for (size_t j = 0; j < output_layer_[i].weights.size(); ++j)
      out_file << output_layer_[i].weights[j] << std::endl;
  }
  out_file << hidden_layers_.size() << std::endl;
  for (size_t i = 0; i < hidden_layers_.size(); ++i) {
    out_file << hidden_layers_[i].size() << std::endl;
    out_file << hidden_layers_[i][0].weights.size() << std::endl;
    for (size_t j = 0; j < hidden_layers_[i].size(); ++j) {
      out_file << hidden_layers_[i][j].bias << std::endl;
      for (size_t k = 0; k < hidden_layers_[i][j].weights.size(); ++k)
        out_file << hidden_layers_[i][j].weights[k] << std::endl;
    }
  }
  out_file.close();
}

} // End of namespace.
