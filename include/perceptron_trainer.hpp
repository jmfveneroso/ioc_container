#ifndef _PERCEPTRON_TRAINER_HPP_
#define _PERCEPTRON_TRAINER_HPP_

#include "neuron_trainer.hpp"

namespace NeuralNetwork {

class PerceptronTrainer : public INeuronTrainer {
  std::shared_ptr<INeuralNetwork> neural_net_;
  std::vector<TrainingCase> training_cases_;
  double learning_rate_;
  double min_error_;

 public:
  PerceptronTrainer(std::shared_ptr<INeuralNetwork>);
  void LoadTrainingCase(TrainingCase);
  // void void LoadTrainingDataFromFile(const char[]);
  void Clear();
  void Train();

  void set_min_error(double min_error) { min_error_ = min_error; }
  void set_learning_rate(double learning_rate) { learning_rate_ = learning_rate; } 
};

}; // End of namespace.

#endif
