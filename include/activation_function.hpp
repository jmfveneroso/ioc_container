#ifndef _ACTIVATION_FUNCTION_HPP_
#define _ACTIVATION_FUNCTION_HPP_

#include <math.h>

namespace NeuralNetwork {

class IActivationFunction {
 public:
  virtual ~IActivationFunction() {}
  virtual double GetResult(const double&) = 0;
};

class Heaviside : public IActivationFunction {
 public:
  double GetResult(const double& x) override {
    return x >= 0 ? 1 : 0;
  }
};

class Sigmoid : public IActivationFunction {
 public:
  double GetResult(const double& x) override {
    return 1 / (1 + exp(-x));
  }
};

}; // End of namespace.

#endif
