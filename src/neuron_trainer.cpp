#include "neural_network.hpp"

namespace NeuralNetwork {

struct TrainingCase {
  std::vector<double> inputs;
  int result;
  TrainingCase(std::vector<double> inputs, int result) 
    : inputs(inputs), result(result) {
  }
};

class INeuronTrainer {
 public:
  virtual void LoadTrainingCase(TrainingCase) = 0;
  // virtual void LoadTrainingDataFromFile(const char[]) = 0;
  virtual void Train() = 0;
};

}; // End of namespace.

#endif
