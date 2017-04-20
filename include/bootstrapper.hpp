#ifndef _BOOTSTRAPPER_HPP_
#define _BOOTSTRAPPER_HPP_

#include "ioc_container.hpp"
#include "neural_network.hpp"

using namespace NeuralNetwork;

namespace {

class Bootstrapper {
 public:
  static void Bootstrap() {
    static IoC::Container& container = IoC::Container::Get();
    container.RegisterType<NeuralNet, NeuralNet>();
  }
};

} // End of namespace.

#endif
