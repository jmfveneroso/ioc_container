#include <iostream>
#include <cassert>
#include "bootstrapper.hpp"

using namespace NeuralNetwork;

int main() {
  Bootstrapper::Bootstrap();

  static IoC::Container& container = IoC::Container::Get();
  container.RegisterInstance<Heaviside, IActivationFunction>();
  container.RegisterInstance<PerceptronTrainer, INeuronTrainer, INeuralNetwork>();

  std::shared_ptr<INeuralNetwork> neural_net     = container.Resolve<INeuralNetwork>();
  std::shared_ptr<INeuronTrainer> neuron_trainer = container.Resolve<INeuronTrainer>();

  // Configuring the neural net constants.
  Configuration config;
  config.threshold = 0.5;
  config.input_layer = Layer(3, Neuron());
  config.output_layer = Layer(1, Neuron());
  config.hidden_layers = std::vector<Layer>();

  neural_net->LoadConfiguration(config);  

  // NAND.
  neuron_trainer->set_learning_rate(0.1);
  neuron_trainer->set_min_error(0);
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 0, 0 }, 1));  
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 0, 1 }, 1));  
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 1, 0 }, 1));  
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 1, 1 }, 0));  

  neuron_trainer->Train();

  assert(1 == neural_net->Calculate({ 1, 0, 0 }));
  assert(1 == neural_net->Calculate({ 1, 0, 1 }));
  assert(1 == neural_net->Calculate({ 1, 1, 0 }));
  assert(0 == neural_net->Calculate({ 1, 1, 1 }));

  // OR.
  neuron_trainer->Clear();  
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 0, 0 }, 0));  
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 0, 1 }, 1));  
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 1, 0 }, 1));  
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 1, 1 }, 1));  

  neuron_trainer->Train();

  assert(0 == neural_net->Calculate({ 1, 0, 0 }));
  assert(1 == neural_net->Calculate({ 1, 0, 1 }));
  assert(1 == neural_net->Calculate({ 1, 1, 0 }));
  assert(1 == neural_net->Calculate({ 1, 1, 1 }));

  // AND.
  neuron_trainer->Clear();  
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 0, 0 }, 0));  
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 0, 1 }, 0));  
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 1, 0 }, 0));  
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 1, 1 }, 1));  

  neuron_trainer->Train();

  assert(0 == neural_net->Calculate({ 1, 0, 0 }));
  assert(0 == neural_net->Calculate({ 1, 0, 1 }));
  assert(0 == neural_net->Calculate({ 1, 1, 0 }));
  assert(1 == neural_net->Calculate({ 1, 1, 1 }));

  // XOR.
  neuron_trainer->Clear();  
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 0, 0 }, 0));  
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 0, 1 }, 1));  
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 1, 0 }, 1));  
  neuron_trainer->LoadTrainingCase(TrainingCase({ 1, 1, 1 }, 0));  

  bool error_caught = false;
  try {
    neuron_trainer->Train();
  } catch (std::runtime_error& e) {
    error_caught = true;
  }
  assert(error_caught);

  return 0;
}
