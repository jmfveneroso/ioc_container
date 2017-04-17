#ifndef _BACKPROPAGATION_TRAINER_HPP_
#define _BACKPROPAGATION_TRAINER_HPP_

#include <memory>
#include "neuron_trainer.hpp"

namespace NeuralNetwork {

class BackpropagationTrainer : public INeuronTrainer {
  std::shared_ptr<INeuralNetwork> neural_net_;
  std::vector<TrainingCase> training_cases_;
  double learning_rate_;
  double min_error_;
  double momentum_ = 0.0001;

  void Init();
  void Backpropagate(TrainingCase&);
  void UpdateWeights();

 public:
  BackpropagationTrainer(std::shared_ptr<INeuralNetwork>);
  void LoadTrainingCase(TrainingCase);
  // void void LoadTrainingDataFromFile(const char[]);
  void Clear();
  void Train();
  double GetResult(std::vector<double>);

  void set_min_error(double min_error) { min_error_ = min_error; }
  void set_learning_rate(double learning_rate) { learning_rate_ = learning_rate; } 
};

} // End of namespace.

#endif
