#include <iostream>
#include <cassert>
#include <math.h>
#include "bootstrapper.hpp"

using namespace NeuralNetwork;

int main() {
  Bootstrapper::Bootstrap();

  static IoC::Container& container = IoC::Container::Get();
  std::shared_ptr<NeuralNet> neural_net = container.Resolve<NeuralNet>();

  neural_net->set_learning_rate(0.25);
  neural_net->SetOutputLayer({ Neuron(0.2, { 0.2, 0.3 }) });
  neural_net->AddHiddenLayer({ Neuron(0.1, { -0.2, 0.1 }), Neuron(0.1, { -0.1, 0.3 }) });

  // // Example found in class slides.
  for (int i = 0; i < 200; ++i) {
    std::cout << neural_net->ToString() << std::endl;
    double value = neural_net->Predict({ 0.1, 0.9 })[0];
    std::cout << "Error (" << i << "): " << 0.9 - value << std::endl; 
    neural_net->Train(TrainingCase({ 0.1, 0.9 }, { 0.9 }));
    neural_net->UpdateWeights();
  }

  // neuron_trainer->set_learning_rate(0.25);
  // neuron_trainer->set_min_error(0.01);
  // neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 0.1, 0.9 }, 0.9));  
  // neuron_trainer->Train();

  // assert(0.01 > fabs(neuron_trainer->GetResult({ 1, 0.1, 0.9 }) - 0.9));

  return 0;
}
