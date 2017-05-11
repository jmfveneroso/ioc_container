#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <sstream>
#include "bootstrapper.hpp"

// Google profiler.
// #define PROFILE
#ifdef PROFILE
#include <gperftools/profiler.h>
#endif

// #define LEARNING_RATE_DECAY
// #define MOMENTUM
// #define WEIGHT_DECAY

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

double GetCrossEntropy(const std::vector<double>& results, TrainingCase& training_case) {
  double cross_entropy = 0;
  for (int i = 0; i < 10; ++i) {
    cross_entropy += training_case.results[i] * log(results[i]) +
                   (1 - training_case.results[i]) * log(1 - results[i]);
  }
  return -cross_entropy;
}

enum Mode {
  GD = 0,
  SGD,
  MINI_BATCH_10,
  MINI_BATCH_50
};

void CreateNeuralNetwork(std::shared_ptr<NeuralNet> neural_net, size_t num_hidden_neurons) {
  size_t num_classes = 10;
  size_t num_features = 784;

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
}

void Run(
  std::shared_ptr<NeuralNet> neural_net, 
  Mnist* mnist,
  Mode mode, 
  double learning_rate, 
  size_t num_hidden_neurons
) {
  // size_t num_training_cases = 5000;
  size_t num_training_cases = 1000;
  size_t num_epochs = 1000;
  CreateNeuralNetwork(neural_net, num_hidden_neurons);
  neural_net->set_learning_rate(learning_rate);

  auto start_time = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < num_epochs; ++i) {
    double total_ce = 0;
    double error = 0;
    for (size_t j = 0; j < num_training_cases; ++j) {
      TrainingCase& training_case = mnist->GetTrainingCase(j);

      // First we must run the model to get the results in order to
      // calculate the errors.
      std::vector<double> results = neural_net->Predict(training_case.inputs);
      neural_net->Train(training_case);

      // total_mse += GetSquaredError(results, training_case);
      total_ce += GetCrossEntropy(results, training_case);

      int digit = GetDigit(results);
      if (digit != GetDigit(training_case.results)) error += 1;

      if (i == 0) continue; 
      if (j == num_training_cases - 1) { // Gradient Descent.
        neural_net->UpdateWeights();
        continue;
      }

      if (mode == SGD) neural_net->UpdateWeights();
      if (mode == MINI_BATCH_10 && j % 10 == 0) neural_net->UpdateWeights();
      if (mode == MINI_BATCH_50 && j % 50 == 0) neural_net->UpdateWeights();
    }

    auto now = std::chrono::high_resolution_clock::now();
    auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
    std::cout << "epoch : " << i + 1 << ", "
              << "cross entropy: " << total_ce / num_training_cases << ", "
              << "prediction error: " << error / num_training_cases << ", "
              << "time: " << int_ms.count() << std::endl; 
  }
}

std::string ModeToStr(Mode mode) {
  switch (mode) {
    case GD: return "GD";
    case SGD: return "SGD";
    case MINI_BATCH_10: return "MINI_BATCH_10";
    case MINI_BATCH_50: return "MINI_BATCH_50";
    default: return "";
  }
}

int main(int argc, char** argv) {
  (void) argv;
#ifdef PROFILE
  ProfilerStart("neural_net.prof");
#endif

  if (argc != 2) {
    std::cout << "Usage: main [directory]" << std::endl;
    return 1;
  }

  Bootstrapper::Bootstrap();
  srand((unsigned int) time(0));

  static IoC::Container& container = IoC::Container::Get();
  Mnist mnist("data_tp1");

  for (int i = 0; i < 4; ++i) {
    double num_hidden_nodes[3] = { 25, 50, 100 };
    for (int j = 0; j < 3; ++j) {

      double learning_rates[3] = { 0.5, 1, 10 };
      for (int k = 0; k < 3; ++k) {
        std::cout << "Mode: " << ModeToStr(static_cast<Mode>(i)) << std::endl;
        std::cout << "Num hidden nodes: " << num_hidden_nodes[j] << std::endl;
        std::cout << "Learning rate: " << learning_rates[k] << std::endl;
        std::cout << "====================" << std::endl;

        std::shared_ptr<NeuralNet> neural_net = container.Resolve<NeuralNet>();
        Run(neural_net, &mnist, static_cast<Mode>(i), learning_rates[k], num_hidden_nodes[j]);

        // std::stringstream ss;
        // ss << argv[1] << "/mode_" << ModeToStr(static_cast<Mode>(i));
        // ss << "_hn_" << num_hidden_nodes[j];
        // ss << "_lr_" << learning_rates[k];
        // ss << "_weights";
        // neural_net->SaveToFile(ss.str());
        // std::cout << "Wrote to " << ss.str() << std::endl;
      }
    }
  }

#ifdef PROFILE
  ProfilerStop();
#endif
  return 0;
}
