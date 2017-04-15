#include <iostream>
#include <cassert>
#include <math.h>
#include "bootstrapper.hpp"

using namespace NeuralNetwork;

int main() {
  Bootstrapper::Bootstrap();

  static IoC::Container& container = IoC::Container::Get();
  container.RegisterInstance<BackpropagationTrainer, INeuronTrainer, INeuralNetwork>();

  std::shared_ptr<INeuralNetwork> neural_net     = container.Resolve<INeuralNetwork>();
  std::shared_ptr<INeuronTrainer> neuron_trainer = container.Resolve<INeuronTrainer>();

  // Configuring the neural net constants.
  Configuration config;
  config.threshold = 0;
  config.input_layer = Layer(3, Neuron());
  config.output_layer = Layer({ Neuron({ 0.2, 0.2, 0.3 }) });
  config.hidden_layers = std::vector<Layer>({ { Neuron({ 0.1, -0.2, 0.1 }), Neuron({ 0.1, -0.1, 0.3 }) } });
  neural_net->LoadConfiguration(config);  

  // Example found in class slides.
  neuron_trainer->set_learning_rate(0.25);
  neuron_trainer->set_min_error(0.01);
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 0.1, 0.9 }, 0.9));  
  neuron_trainer->Train();

  assert(0.01 > fabs(neuron_trainer->GetResult({ 1, 0.1, 0.9 }) - 0.9));

  // XOR.
  // neuron_trainer->set_learning_rate(0.25);
  // neuron_trainer->set_min_error(0.01);
  // neuron_trainer->Clear();
  // neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 0, 0 }, 0));  
  // neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 0, 1 }, 1));  
  // neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 1, 0 }, 1));  
  // neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 1, 1 }, 0));  
  // neuron_trainer->Train();

  // assert(0.01 > fabs(neuron_trainer->GetResult({ 1, 0, 0 }) - 0));
  // assert(0.01 > fabs(neuron_trainer->GetResult({ 1, 0, 1 }) - 1));
  // assert(0.01 > fabs(neuron_trainer->GetResult({ 1, 1, 0 }) - 1));
  // assert(0.01 > fabs(neuron_trainer->GetResult({ 1, 1, 1 }) - 0));
  return 0;
}
