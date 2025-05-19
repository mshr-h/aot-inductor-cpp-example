#include <iostream>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

int main() {
    c10::InferenceMode mode;

    torch::inductor::AOTIModelPackageLoader loader("model.pt2");
    std::vector<torch::Tensor> inputs = {torch::randn({1, 3, 224, 224}, at::kCUDA)};
    std::vector<torch::Tensor> outputs = loader.run(inputs);
    std::cout << "Result from the inference:"<< std::endl;
    std::cout << outputs << std::endl;

    return 0;
}
