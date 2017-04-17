#include "perceptron_trainer.hpp"
#include <iostream>

namespace NeuralNetwork {

PerceptronTrainer::PerceptronTrainer(std::shared_ptr<INeuralNetwork> neural_net) 
  : neural_net_(neural_net) {
}

void PerceptronTrainer::LoadTrainingCase(TrainingCase training_case) {
  training_cases_.push_back(training_case);
}

void PerceptronTrainer::Clear() {
  training_cases_ = std::vector<TrainingCase>();
}

double PerceptronTrainer::GetResult(std::vector<double> inputs) {
  return (neural_net_->Calculate(inputs) > 0.5) ? 1 : 0;
}

void PerceptronTrainer::Train() {
  double min_achieved_error = 0;
  int error_reduction_counter = 0;

  double error = 1;
  while (error > min_error_) { 
    error = 0;
    for (size_t i = 0; i < training_cases_.size(); ++i) {
      double& expected_result = training_cases_[i].result;
      double result = GetResult(training_cases_[i].inputs);

      if (result == expected_result) continue;
    
      error += 1;

      // Change the weights.
      Layer& output_layer = neural_net_->GetOutputLayer();
      for (size_t j = 0; j < output_layer.size(); ++j) {
        Neuron& neuron = output_layer[j];
        for (size_t k = 0; k < neuron.weights.size(); ++k) {
          neuron.weights[k] += learning_rate_ * (expected_result - result) * 
                               training_cases_[i].inputs[k];
        }
      }
    }
    error = error / training_cases_.size();
    if (error < min_achieved_error) {
      min_achieved_error = error;
      error_reduction_counter = 0;
    } else {
      ++error_reduction_counter;
    }

    // After 1000 runs, the error has not reduced.
    if (error_reduction_counter > 1000) 
      throw std::runtime_error("The problem is not linearly separable.");
  }
}

} // End of namespace.
