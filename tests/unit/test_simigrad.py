import numpy as np
import deepspeed
import pytest
from deepspeed.runtime.simigrad import SimiGrad
from common import distributed_test
from simple_model import SimpleModel, SimpleOptimizer, random_dataloader, args_from_dict
import torch
import time

# @distributed_test(world_size=[4])
# def test_simigrad_fp16():

#     config_dict = {
#         "train_batch_size": 4,
#         "steps_per_print": 1,
#         "optimizer": {
#             "type": 'Adam',
#             "params": {
#                 "lr": 0.0001
#             }
#         },
#         "gradient_clipping": 1.0,
#         "fp16": {
#             "enabled": True
#         },
#         "simigrad": {
#             "enabled": True
#         }
#     }

#     args = args_from_dict(".", config_dict)
#     time.sleep(0.1)
#     hidden_dim = 10

#     model = SimpleModel(hidden_dim, empty_grad=False)

#     model, _, _, _ = deepspeed.initialize(args=args,
#                                           model=model,
#                                           model_parameters=model.parameters())
#     assert model.simigrad_enabled()
#     data_loader = random_dataloader(model=model,
#                                     total_samples=50,
#                                     hidden_dim=hidden_dim,
#                                     device=model.device)
#     for i, batch in enumerate(data_loader):
#         model._config.simigrad_enabled = False
#         loss = model(batch[0], batch[1])
#         # print(i,loss)
#         model.backward(loss)
#         grads_without_simigrad = model.simigrad.get_optimizer_grad(model)
#         # print(grads_without_simigrad)
#         model.zero_grad()


#         model._config.simigrad_enabled = True
#         loss = model(batch[0], batch[1])
#         model.backward(loss)
#         grads_with_simigrad = model.simigrad.get_optimizer_grad(model)



#         for grad1, grad2 in zip(grads_with_simigrad, grads_without_simigrad):
#             try:
#                 assert torch.allclose(grad1[torch.isfinite(grad1)], grad2[torch.isfinite(grad2)],rtol=0.1,atol=1e-5)
#             except Exception as e:
#                 if torch.distributed.get_rank()==0:
#                     a=torch.isclose(grad1, grad2,rtol=0.1,atol=1e-5,equal_nan=True)
#                     print(grad1[~a],grad2[~a])
#                 raise e

#         model.step()



@distributed_test(world_size=[4])
def test_simigrad_fp32():

    config_dict = {
        "train_batch_size": 4,
        "steps_per_print": 1,
        "optimizer": {
            "type": 'Adam',
            "params": {
                "lr": 0.0001
            }
        },
        "fp16": {
            "enabled": False
        },
        "simigrad": {
            "enabled": True
        }
    }

    args = args_from_dict(".", config_dict)
    time.sleep(0.1)
    hidden_dim = 10

    model = SimpleModel(hidden_dim, empty_grad=False)

    model, _, _, _ = deepspeed.initialize(args=args,
                                          model=model,
                                          model_parameters=model.parameters())
    assert model.simigrad_enabled()
    data_loader = random_dataloader(model=model,
                                    total_samples=50,
                                    hidden_dim=hidden_dim,
                                    device=model.device,
                                    dtype=torch.float)
    for i, batch in enumerate(data_loader):
        model._config.simigrad_enabled = False
        loss = model(batch[0], batch[1])
        print(i,loss)
        model.backward(loss)
        for param_name, param in model.module.named_parameters():
            print(param.grad.data)
        grads_without_simigrad = model.simigrad.get_optimizer_grad(model)
        model.zero_grad()


        model._config.simigrad_enabled = True
        loss = model(batch[0], batch[1])
        model.backward(loss)
        grads_with_simigrad = model.simigrad.get_optimizer_grad(model)
        # model.zero_grad()

        # assert torch.all(torch.isfinite(loss))
        for grad1, grad2 in zip(grads_with_simigrad, grads_without_simigrad):
            assert torch.allclose(grad1, grad2,atol=1e-5), [grad1, grad2]

        model.step()
