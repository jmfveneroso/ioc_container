#include "backpropagation_trainer.hpp"
#include <iostream>
#include <math.h>

namespace NeuralNetwork {

BackpropagationTrainer::BackpropagationTrainer(
  std::shared_ptr<INeuralNetwork> neural_net
) : neural_net_(neural_net) {
}

void BackpropagationTrainer::LoadTrainingCase(TrainingCase training_case) {
  training_cases_.push_back(training_case);
}

void BackpropagationTrainer::Clear() {
  training_cases_ = std::vector<TrainingCase>();
}

void BackpropagationTrainer::Init() {
  Layer& input_layer = neural_net_->GetInputLayer();
  for (size_t i = 0; i < input_layer.size(); ++i) {
    input_layer[i].Clear();
  }
  for (size_t i = 0; i < neural_net_->GetNumHiddenLayers(); ++i) {
    Layer hidden_layer = neural_net_->GetHiddenLayer(i);
    for (size_t j = 0; j < hidden_layer.size(); ++j) {
      hidden_layer[i].Clear();
    }
  }
  Layer& output_layer = neural_net_->GetOutputLayer();
  for (size_t i = 0; i < output_layer.size(); ++i) {
    output_layer[i].Clear();
  }
}

void BackpropagationTrainer::Backpropagate(TrainingCase& training_case) {
  GetResult(training_case.inputs);

  Layer* current_layer = &neural_net_->GetOutputLayer();
  Layer* prev_layer;

  std::vector<double> output_derivatives = std::vector<double>(current_layer->size(), 0);
  std::vector<double> prev_layer_output_derivatives;

  for (size_t i = 0; i < current_layer->size(); ++i)
    output_derivatives[i] = -(training_case.result - (*current_layer)[i].result);
  
  size_t prev_layer_index = 0;
  while (prev_layer_index <= neural_net_->GetNumHiddenLayers()) {
    size_t prev_layer_size = 0;
    bool is_input = false;
    if (prev_layer_index == neural_net_->GetNumHiddenLayers()) {
      prev_layer = &neural_net_->GetInputLayer();
      is_input = true;
      prev_layer_size = prev_layer->size() - 1;
    } else {
      prev_layer = &neural_net_->GetHiddenLayer(prev_layer_index);
      prev_layer_size = prev_layer->size();
    }

    prev_layer_output_derivatives = std::vector<double>(prev_layer->size(), 0);
   
    for (size_t i = 0; i < current_layer->size(); ++i) {
      Neuron& neuron = (*current_layer)[i];
      double total_input_derivative = neuron.result * (1 - neuron.result) * output_derivatives[i];

      neuron.weight_derivatives[0] += 1 * total_input_derivative;
 
      for (size_t j = 0; j < prev_layer_size; ++j) {
        size_t index = (is_input) ? j + 1 : j;
        Neuron& hidden_neuron = (*prev_layer)[index];
        neuron.weight_derivatives[j + 1] += hidden_neuron.result * total_input_derivative;
        prev_layer_output_derivatives[j] += neuron.weights[j + 1] * total_input_derivative;
      }
    }
 
    current_layer = prev_layer;
    output_derivatives = prev_layer_output_derivatives;
    ++prev_layer_index;
  }
}

void BackpropagationTrainer::UpdateWeights() {
  Layer& output_layer = neural_net_->GetOutputLayer();
  for (size_t i = 0; i < output_layer.size(); ++i) {
    Neuron& neuron = output_layer[i];
    for (size_t j = 0; j < neuron.weights.size(); ++j) {
      neuron.weights[j] += learning_rate_ * -neuron.weight_derivatives[j] + momentum_ * neuron.weights[j];
    }
  }
  for (size_t i = 0; i < neural_net_->GetNumHiddenLayers(); ++i) {
    Layer& hidden_layer = neural_net_->GetHiddenLayer(i);
    for (size_t j = 0; j < hidden_layer.size(); ++j) {
      Neuron& neuron = hidden_layer[j];
      for (size_t k = 0; k < neuron.weights.size(); ++k) {
        neuron.weights[k] += learning_rate_ * (-neuron.weight_derivatives[k]) + momentum_ * neuron.weights[k];
      }
    }
  }
}

void BackpropagationTrainer::Train() {
  int counter = 0;
  double error = 1;
  while (error > min_error_) { 
    error = 0;
    std::cout << neural_net_->ToString() << std::endl;
    // Clear weight derivatives.
    Init();

    for (size_t i = 0; i < training_cases_.size(); ++i) {
      Backpropagate(training_cases_[i]);
    }
    UpdateWeights();
    std::cout << neural_net_->ToString() << std::endl;
    for (size_t i = 0; i < training_cases_.size(); ++i) {
      error += fabs(training_cases_[i].result - GetResult(training_cases_[i].inputs));
    }
    error /= training_cases_.size();
    std::cout << ++counter << " error: " << error << std::endl;
  }
}

double BackpropagationTrainer::GetResult(std::vector<double> inputs) {
  return neural_net_->Calculate(inputs);
}

} // End of namespace.
