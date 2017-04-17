#ifndef _BOOTSTRAPPER_HPP_
#define _BOOTSTRAPPER_HPP_

#include "ioc_container.hpp"
#include "perceptron_trainer.hpp"
#include "backpropagation_trainer.hpp"

using namespace NeuralNetwork;

namespace {

class Bootstrapper {
 public:
  static void Bootstrap() {
    static IoC::Container& container = IoC::Container::Get();
    container.RegisterInstance<NeuralNet, INeuralNetwork>();
    container.RegisterInstance<PerceptronTrainer, INeuronTrainer, INeuralNetwork>();
  }
};

} // End of namespace.

#endif
