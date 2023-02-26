import torch 

if torch.cuda.is_available():
    device_ = torch.device("cuda:0")
    print(f"properties{torch.cuda.get_device_properties(device_)}\n")
    print(f"memory {torch.cuda.mem_get_info(device_)}\n")
    print(torch.version.cuda)
    print(torch.)
