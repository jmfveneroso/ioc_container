#ifndef _NEURON_TRAINER_HPP_
#define _NEURON_TRAINER_HPP_

#include "neural_network.hpp"

namespace NeuralNetwork {

struct TrainingCase {
  std::vector<double> inputs;
  double result;
  TrainingCase(std::vector<double> inputs, double result) 
    : inputs(inputs), result(result) {
  }
};

class INeuronTrainer {
 public:
  virtual void LoadTrainingCase(TrainingCase) = 0;
  // virtual void LoadTrainingDataFromFile(const char[]) = 0;
  virtual void Clear() = 0;
  virtual void Train() = 0;
  virtual double GetResult(std::vector<double>) = 0;
  virtual void set_min_error(double learning_rate) = 0;
  virtual void set_learning_rate(double min_error) = 0;
};

} // End of namespace.

#endif
