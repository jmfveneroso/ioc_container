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

double NeuralNet::CalculateNeuron(Neuron& neuron, std::vector<double> inputs) {
  neuron.result = neuron.bias;
  for (size_t i = 0; i < inputs.size(); ++i) 
    neuron.result += neuron.weights[i] * inputs[i];
  return neuron.result = ActivationFunction(neuron.result);
}

std::vector<double> NeuralNet::Predict(std::vector<double> inputs) {
  // Feed the hidden layers.
  std::vector<double> results;
  for (size_t i = 0; i < hidden_layers_.size(); ++i) {
    results = std::vector<double>(hidden_layers_[i].size());
    for (size_t j = 0; j < hidden_layers_[i].size(); ++j)
      results[j] = CalculateNeuron(hidden_layers_[i][j], inputs);
    inputs = results; 
  }

  // Feed the output layer.
  results = std::vector<double>(output_layer_.size());
  for (size_t i = 0; i < output_layer_.size(); ++i)
    results[i] = CalculateNeuron(output_layer_[i], inputs);
     
  return results;
}

void NeuralNet::Train(const TrainingCase& training_case) {
  // First we must run the model to get the results in order to
  //  calculate the errors.
  Predict(training_case.inputs);

  for (size_t i = 0; i < output_layer_.size(); ++i) {
    Neuron& neuron = output_layer_[i];

    // We are computing the inverse of the gradient, so no need for the minus 
    // sign here. Consequently, it will also be omitted when updating weights.
    double error = training_case.results[i] - neuron.result;
    neuron.delta = neuron.result * (1 - neuron.result) * error;
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
        prev_neuron.weight_derivatives[j] = prev_neuron.delta * neuron.result;
      }
      neuron.delta = neuron.result * (1 - neuron.result) * output_derivative;
    }
    prev_layer = &hidden_layers_[i];
  }

  for (size_t i = 0; i < prev_layer->size(); ++i) {
    Neuron& neuron = (*prev_layer)[i];
    for (size_t j = 0; j < training_case.inputs.size(); ++j) {
      neuron.weight_derivatives[j] = neuron.delta * training_case.inputs[j];
    }
  }
}

void NeuralNet::UpdateNeuron(Neuron& neuron) {
  neuron.bias += learning_rate_ * neuron.delta + momentum_ * neuron.bias;
  for (size_t k = 0; k < neuron.weights.size(); ++k)
    neuron.weights[k] += learning_rate_ * neuron.weight_derivatives[k] + 
                         momentum_ * neuron.weights[k];
}

void NeuralNet::UpdateWeights() {
  for (size_t i = 0; i < hidden_layers_.size(); ++i)
    for (size_t j = 0; j < hidden_layers_[i].size(); ++j)
      UpdateNeuron(hidden_layers_[i][j]);

  for (size_t i = 0; i < output_layer_.size(); ++i)
    UpdateNeuron(output_layer_[i]);
}

std::string NeuralNet::NeuronToString(Neuron& neuron) {
  std::stringstream ss;
  ss << "    Neuron " << neuron.id << std::endl
     << "      bias: " << neuron.bias << std::endl
     << "      weights: " << std::endl;
  for (size_t i = 0; i < neuron.weight_derivatives.size(); ++i) {
    ss << "        " << i << " value: " << neuron.weights[i];
    ss << ", derivative: " << neuron.weight_derivatives[i] << std::endl;
  }
  ss << "      result: " << neuron.result << std::endl;
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
