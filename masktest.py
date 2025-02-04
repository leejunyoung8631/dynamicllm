import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import gumbel_softmax
from torch.nn.parameter import Parameter
import torch.optim as optim

# from megatron import get_args
# from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, get_tensor_model_parallel_world_size, _initialize_affine_weight_cpu, _initialize_affine_weight_gpu, linear_with_grad_accumulation_and_async_allreduce, set_tensor_model_parallel_attributes, _grad_accum_fusion_available, linear_with_frozen_weight
# from megatron.core.tensor_parallel.utils import (
#     divide,
# )
# from megatron.core.model_parallel_config import ModelParallelConfig
# from megatron.core.parallel_state import (
#     get_tensor_model_parallel_world_size,
# )
# from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
# from megatron.core.tensor_parallel.mappings import (
#     copy_to_tensor_model_parallel_region,
#     gather_from_tensor_model_parallel_region,
#     reduce_from_tensor_model_parallel_region,
#     reduce_scatter_to_sequence_parallel_region,
#     scatter_to_tensor_model_parallel_region,
# )
# from megatron.core.tensor_parallel.utils import divide
# from megatron import print_rank_0

# from megatron.model.utils import init_method_normal, scaled_init_method_normal


from functools import partial
class DifferentiableMask(nn.Module):
    def __init__(
            self, 
            target_param, 
            N=2, 
            M=4, 
            hard=False,
            temperature=3.0, 
            scale_multiplier=[1e3, 1e4], 
        ):
        '''
        Implementation of differantiable mask learner
        args:
            temperature: temperature of the gumbel distribution
            gate_param: the parameter to be masked
            init_prob: initial probability of the mask
            scale_multiplier: multiplies learned gates by this value, it is needed to make the gates more sensitive to small learning rates
            initialization: "none" means we start from 0.95 for all and dont bother with initialization, "initial_mask" means we start from the initial mask
            hard: sampling mask for full gumbel
            
        temperature parameter needs more attention, we should do annealing it during training
        '''
        super().__init__()
        
        
        # simple setting
        N=2 
        M=4 
        hard=False 
        temperature=[4, 0.1]
        scale_multiplier=[1e3, 1e4]
        freeze_weight=False
        freeze_mask=False
        
        
        self.N = N
        self.M = M
        # self.args = get_args() # for scheduling
        self.mask_difference = 1.0
        self.initial_mask_size = target_param.size()
        self.temperature = temperature
        self.scale_multiplier = scale_multiplier
        
        # Gumbel parameters
        init_partial = partial(init.normal_, std=0.01)
        
        
        print("Init mask options")
        # self.gate = Parameter(torch.empty(
        #         128, 11,
        #         device=torch.cuda.current_device(), dtype=torch.float32))
        # torch.nn.init.normal_(self.gate, std=0.01)
        index = torch.arange(11).unsqueeze(0)
        mask_options = self.make_mask(index).cuda()
        print("mask option")
        print(mask_options.shape)
        print(mask_options)
        
        

        # if self.N==2 and self.M==4:
        #     print("initalizing mask options for 2:4...")
        #     # self.gate = Parameter(torch.empty(
        #     #     target_param.numel()//4, 6,
        #     #     device=torch.cuda.current_device(), dtype=torch.float32))
        #     self.gate = Parameter(torch.empty(
        #         128, 10,
        #         device=torch.cuda.current_device(), dtype=torch.float32))
        #     torch.nn.init.normal_(self.gate, std=0.01)
        #     # _initialize_affine_weight_gpu(self.gate, init_partial, partition_dim=1, stride=1)
        #     # mask_options = torch.zeros(1, 6, 4, 
        #                             # device=torch.cuda.current_device(), dtype=torch.float32)
        #     mask_options = torch.zeros(1, 10, 4, 
        #                             device=torch.cuda.current_device(), dtype=torch.float32)
        #     mask_options[:, 0, :].data += torch.tensor([1, 1, 0, 0], device=torch.cuda.current_device(), dtype=torch.float32)
        #     mask_options[:, 1, :].data += torch.tensor([1, 0, 1, 0], device=torch.cuda.current_device(), dtype=torch.float32)
        #     mask_options[:, 2, :].data += torch.tensor([1, 0, 0, 1], device=torch.cuda.current_device(), dtype=torch.float32)
        #     mask_options[:, 3, :].data += torch.tensor([0, 1, 1, 0], device=torch.cuda.current_device(), dtype=torch.float32)
        #     mask_options[:, 4, :].data += torch.tensor([0, 1, 0, 1], device=torch.cuda.current_device(), dtype=torch.float32)
        #     mask_options[:, 5, :].data += torch.tensor([0, 0, 1, 1], device=torch.cuda.current_device(), dtype=torch.float32)
        #     mask_options[:, 6, :].data += torch.tensor([1, 0, 1, 0], device=torch.cuda.current_device(), dtype=torch.float32)
        #     mask_options[:, 7, :].data += torch.tensor([1, 0, 0, 1], device=torch.cuda.current_device(), dtype=torch.float32)
        #     mask_options[:, 8, :].data += torch.tensor([0, 1, 1, 0], device=torch.cuda.current_device(), dtype=torch.float32)
        #     mask_options[:, 9, :].data += torch.tensor([0, 1, 0, 1], device=torch.cuda.current_device(), dtype=torch.float32)
        # elif self.N==1 and self.M==4:
        #     print("initalizing mask options for 1:4...")
        #     self.gate = Parameter(torch.empty(
        #         128, 7,
        #         device=torch.cuda.current_device(), dtype=torch.float32))
        #     torch.nn.init.normal_(self.gate, std=0.01)
        #     # _initialize_affine_weight_gpu(self.gate, init_partial, partition_dim=1, stride=1)
        #     mask_options = torch.zeros(1, 4, 4, 
        #                             device=torch.cuda.current_device(), dtype=torch.float32)
        #     mask_options[:, 0, :].data += torch.tensor([1, 0, 0, 0], device=torch.cuda.current_device(), dtype=torch.float32)
        #     mask_options[:, 1, :].data += torch.tensor([0, 1, 0, 0], device=torch.cuda.current_device(), dtype=torch.float32)
        #     mask_options[:, 2, :].data += torch.tensor([0, 0, 1, 0], device=torch.cuda.current_device(), dtype=torch.float32)
        #     mask_options[:, 3, :].data += torch.tensor([0, 0, 0, 1], device=torch.cuda.current_device(), dtype=torch.float32)
        # else:
        #     raise NotImplementedError
        
        
        self.register_buffer("mask_options", mask_options)
        self.hard = hard

        self.current_scale_multiplier = self.scale_multiplier[0]
        self.current_temperature = self.temperature[0]
        self.current_max_prob = 0.0
        
        
        # temp for debug
        self.whole_iter = 10000
        self.iteration = 1
        self.training = True
        
    
    
    def make_mask(self, index):
        # _, index = torch.max(index, dim=-1)
        
        token_mask = None
        order = [24, 26, 25, 10, 27, 13, 22, 14, 9, 29, 12]
        b, n = index.shape
        token_mask = torch.ones((b, n, 32))
        for d in range(b):
            for t in range(n):
                n_zero = index[d, t]
                token_mask[d, t, order[:n_zero]] = 0
        # token_mask = token_mask.permute(0, 2, 1)
        
        return token_mask
    
    
    

    def forward(self, x): 
        self.gate = x
        print(self.gate.shape)
        
        if self.training:
            start_temp, end_temp = self.temperature 
            # self.current_temperature = start_temp + (end_temp - start_temp) * (self.args.iteration / self.args.train_iters)
            self.current_temperature = start_temp + (end_temp - start_temp) * (self.iteration / self.whole_iter)
            start_scale, end_scale = self.scale_multiplier
            self.current_scale_multiplier = start_scale + (end_scale - start_scale) * (self.iteration / self.whole_iter)
            
            print("current temperature")
            print(self.current_temperature)
            print("current multiplier")
            print(self.current_scale_multiplier)
            
            sampling_tensor = self.gate * self.current_scale_multiplier
            print("sampling_tensor")
            print(sampling_tensor.shape)
            
            choices = gumbel_softmax(sampling_tensor, tau=self.current_temperature, hard=self.hard)
            print("gumbel_softmax")
            print(choices.shape)
            
            self.current_max_prob = choices.max(-1)[0].mean().item()
            print("current_max_prob")
            print(self.current_max_prob)
            
            # backprop_gate = (choices.unsqueeze(1) @ self.mask_options).squeeze(1)
            backprop_gate = (choices @ self.mask_options).squeeze(1)
            print(choices.shape)
            print(self.mask_options.shape)
            print("backprop_gate")
            print(backprop_gate.shape)
            

            
            # backprop_gate = backprop_gate.reshape(self.initial_mask_size)
        else:
            # just based on the maximum logit
            backprop_gate = self.mask_options[torch.arange(self.mask_options.shape[0]), self.gate.argmax(dim=-1)]
            backprop_gate = backprop_gate.reshape(self.initial_mask_size)
        self.sampled_gate = backprop_gate
        
        
        # temp code
        self.iteration += 1
        
        return backprop_gate
    


class MLP(nn.Module):
    def __init__(self ):
        super().__init__()
        self.mlp = nn.Linear(4096, 11)
    
    def forward(self, x):
        x = self.mlp(x)
        
        return x



def main():
    # reday
    mlp = MLP().cuda()
    
    
    token = torch.rand((4, 12, 4096), dtype=torch.float).cuda() # b, n, d
    feature = mlp(token) # b, n, 11
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    
    
    print("feature")
    print(feature.shape)
    
    back_mask = DifferentiableMask(feature)
    bp = back_mask(feature)
    
    bp = bp[0]
    for i in range(10):
        print(bp[i])
    
    v_loss = torch.sum(bp, dim=-1)
    print(v_loss)
    logits = torch.mean(v_loss)
    
    optimizer.zero_grad()
    logits.backward()
    optimizer.step()
    
    print("done backprobagataion")
    
    
    
    return 0





if __name__ == "__main__":
    main()