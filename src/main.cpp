#include <iostream>
#include <fstream>
#include "bootstrapper.hpp"

class Mnist {
  TrainingCase training_cases_[5000];

  void Load(const std::string& filename) {
    std::ifstream in_file(filename);
    if (!in_file.is_open()) 
      throw new std::runtime_error("File \"" + filename + "\" not found.");

    for (size_t i = 0; i < 5000; ++i) {
      training_cases_[i].results = std::vector<double>(10, 0);
      training_cases_[i].inputs = std::vector<double>(784, 0);

      size_t digit;
      in_file >> digit;
      training_cases_[i].results[digit] = 1;

      for (size_t j = 0; j < 784; ++j) { 
        char c;
        in_file >> c;
        in_file >> training_cases_[i].inputs[j]; 
      }
    }
  }

 public:
  Mnist(const std::string& filename) {
    Load(filename);
  }

  TrainingCase& GetTrainingCase(size_t i) { return training_cases_[i]; }
};

struct GenerateRandom { 
  double operator()() { 
    return (rand() / (double) RAND_MAX) * 8 - 4;
  }
};

int GetDigit(std::vector<double> results) {
  double max = results[0];
  int pos = 0;
  for (size_t i = 1; i < 10; ++i) {
    if (results[i] > max) {
      max = results[i];
      pos = i;
    }
  }
  return pos;
}

double GetSquaredError(const std::vector<double>& results, TrainingCase& training_case) {
  double squared_error = 0;
  for (int i = 0; i < 10; ++i)
    squared_error += pow(results[i] - training_case.results[i], 2);
  return 0.5 * squared_error;
}

int main () {
  Bootstrapper::Bootstrap();
  static IoC::Container& container = IoC::Container::Get();
  std::shared_ptr<NeuralNet> neural_net = container.Resolve<NeuralNet>();

  srand((unsigned int) time(0));

  Mnist mnist("data_tp1");

  neural_net->set_learning_rate(0.5);

  size_t num_hidden_neurons = 100;
  size_t num_classes = 10;
  size_t num_features = 784;
  size_t num_training_cases = 200;

  // Output Layer.
  Layer output_layer;
  for (size_t i = 0; i < num_classes; ++i) {
    std::vector<double> weights(num_hidden_neurons);
    std::generate_n(weights.begin(), num_hidden_neurons, GenerateRandom());
    output_layer.push_back(Neuron(GenerateRandom()(), weights));
  }
  neural_net->SetOutputLayer(output_layer);

  Layer hidden_layer;
  for (size_t i = 0; i < num_hidden_neurons; ++i) {
    std::vector<double> weights(num_features);
    std::generate_n(weights.begin(), num_features, GenerateRandom());
    hidden_layer.push_back(Neuron(GenerateRandom()(), weights));
  }
  neural_net->AddHiddenLayer(hidden_layer);

  for (size_t i = 0; i < 10000; ++i) {
    double total_mse = 0;
    double error = 0;
    for (size_t j = 0; j < num_training_cases; ++j) {
      TrainingCase& training_case = mnist.GetTrainingCase(j);
      // First we must run the model to get the results in order to
      // calculate the errors.
      std::vector<double> results = neural_net->Predict(training_case.inputs);
      neural_net->Train(training_case);

      total_mse += GetSquaredError(results, training_case);

      int digit = GetDigit(results);
      if (digit != GetDigit(training_case.results)) error += 1;
    }
    neural_net->UpdateWeights();
    // std::cout << neural_net->ToString() << std::endl;
    std::cout << "mse: " << total_mse / num_training_cases << ", prediction error: " 
              << error / num_training_cases << std::endl; 
  }

  // std::cout << neural_net->ToString();

  return 0;
}
