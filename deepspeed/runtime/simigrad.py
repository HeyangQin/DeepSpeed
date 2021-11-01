from logging import log
from deepspeed.utils import log_dist
from collections import defaultdict
import torch
import time
from torch.autograd.grad_mode import F
import torch.distributed as dist
import math
import numpy as np
from torch._utils import _flatten_dense_tensors
from torch.nn.parallel import distributed


# def merge_grad(parameters):
#     if isinstance(parameters, torch.Tensor):
#         parameters = [parameters]
#     if len(parameters) == 0:
#         return torch.tensor(0.)
#     total_norm = torch.cat([p.detach().view(-1) for p in parameters])
#     return total_norm


def create_two_half_groups_from_list(ranks_list):
    ranks_list=list(ranks_list)
    if len(ranks_list)<=1:
        return None
    current_group=None
    first_half=ranks_list[:len(ranks_list)//2]
    if len(first_half)>1:
        log_dist(f"SimiGrad creating group={first_half}",ranks=[0])
        group = dist.new_group(first_half)
        if dist.get_rank() in first_half:
            current_group=group
    second_half=ranks_list[len(ranks_list)//2:]
    if len(second_half)>1:
        log_dist(f"SimiGrad creating group={second_half}",ranks=[0])
        group = dist.new_group(second_half)
        if dist.get_rank() in second_half:
            current_group=group
    return current_group


def get_all_ranks_from_parallel_group(group):
    rank=0
    results=[]
    try:
        while True:
            results.append(dist.distributed_c10d._get_global_rank(group, rank))
            rank+=1
    except RuntimeError:
        pass
    return results


class NoAvailGradError(Exception):
    """No valid gradient is found. Usually due to overflow"""
    pass

class SimiGrad(object):
    def __init__(self, engine):
        super().__init__()

        self.engine = engine
        self.params = defaultdict(
            lambda: None, self.engine._config.simigrad_params)
        # assert self.engine.mpu is None and not self.pipeline_parallelism, "SimiGrad doesn't work with pipeline parallelism at this point."
        # assert self.params["enable_adjust"] is not None, "SimiGrad is enabled yet "
        assert dist.get_world_size() > 1, "Currently SimiGrad needs at least 2 GPUs to work. SimiGrad with a single GPU is still under developemnt."

        assert self.params["batch_size_upper_bound"] is None or self.params["batch_size_upper_bound"] >= 1
        assert self.params["batch_size_lower_bound"] is None or self.params["batch_size_lower_bound"] >= 1
        if self.params["batch_size_lower_bound"] is not None and self.params["batch_size_upper_bound"] is not None:
            assert self.params["batch_size_upper_bound"] >= self.params["batch_size_lower_bound"]
        self.params["global_lr_modifier"] = 1.0
        self.gradient_accumulation_offset = 0
        self.gradient_step_size = max(
            int(self.engine.gradient_accumulation_steps()/10), 1)
        self.params["original_batch_size"] = self.engine.train_batch_size()
        self.params["batch_size_similiarity_record"] = dict()
        self.params["similarity_history"] = []
        self.gradient_accumulation_start = 0
        self.dp_group=None
        self.grad_passing_source=None

        if not self.engine.mpu: #data parallelism only
            world_size = dist.get_world_size()
            self.dp_group=create_two_half_groups_from_list(range(world_size))
            self.grad_passing_group=dist.new_group([0, world_size-1])
            if dist.get_rank() in [0, world_size-1]:
                self.grad_passing_source=world_size-1
            else:
                self.grad_passing_source=None
        else:
            # # According to mpu:
            # # Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
            # # use 2 GPUs to parallelize the model. The present function will
            # # create 4 model parallel groups and 2 data parallel grous as:
            # #     4 model parallel groups:
            # #         [g0, g1], [g2, g3], [g4, g5], [g6, g7]
            # #     2 data parallel groups:
            # #         [g0, g2, g4, g6], [g1, g3, g5, g7]
            # world_size = dist.get_world_size()
            # model_parallel_size=self.engine.mpu.get_model_parallel_world_size()
            # assert world_size>=model_parallel_size*2, "SimiGrad currently cannot work with model parallelism without data parallelism."
            # for i in range(model_parallel_size):
            #     ranks = list(range(i, world_size, model_parallel_size))
            #     group=create_two_half_groups_from_list(ranks)
            #     self.dp_group=group if self.dp_group is None else self.dp_group
            #     grad_passing_group=dist.new_group([ranks[0],ranks[-1]])
            #     log_dist(f"SimiGrad created grad passing group for ranks {[ranks[0],ranks[-1]]}",ranks=[-1])
            #     if dist.get_rank() in [ranks[0],ranks[-1]]:
            #         self.grad_passing_source=ranks[-1]
            #         self.grad_passing_group=grad_passing_group
            
            #experimental code
            world_size = dist.get_world_size()
            known_replicas=[]
            for target_rank in range(world_size):
                if target_rank in known_replicas:
                    continue
                if dist.get_rank()==target_rank:
                    local_replicas=get_all_ranks_from_parallel_group(self.engine.mpu.get_data_parallel_group())
                    # log_dist(f"My current{local_replicas}",ranks=[-1])
                    dist.broadcast(torch.Tensor(local_replicas).cuda().int(),dist.get_rank())
                else:
                    local_replicas=torch.Tensor(self.engine.mpu.get_data_parallel_world_size()).cuda().int()
                    dist.broadcast(local_replicas,target_rank)
                    local_replicas=local_replicas.tolist()
                group=create_two_half_groups_from_list(local_replicas)
                self.dp_group=group if self.dp_group is None else self.dp_group
                grad_passing_group=dist.new_group([local_replicas[0],local_replicas[-1]])
                log_dist(f"SimiGrad created grad passing group for ranks {[local_replicas[0],local_replicas[-1]]}",ranks=[0])
                if dist.get_rank() in [local_replicas[0],local_replicas[-1]]:
                    self.grad_passing_source=local_replicas[-1]
                    self.grad_passing_group=grad_passing_group
                known_replicas+=local_replicas
                log_dist(f"So far SimiGrad has crated groups for ranks:{known_replicas}",ranks=[0])

        log_dist(f"SimiGrad grad passing source:{self.grad_passing_source}",ranks=[-1])
            
        # need to be initialized later because the model may be not initalized at this point
        self.grad_placeholder = None
        self.cos_placeholder = torch.rand(1).cuda()
        self.adjust_direction = 0

        log_dist(f'Enabled SimiGrad ({self.params})', ranks=[0])

    def print_cosine_similarity(self,ranks=[0]):
        log_dist(f"cosine {self.cos_placeholder.item()} computation at micro step {self.engine.micro_steps} takes {self.cosine_computation_time:.4f}s", ranks=ranks)


    def get_cosine_similarity(self):
        try:
            if self.params["adjust_interval"] is not None and self.engine.global_steps % self.params["adjust_interval"] != 0:
                log_dist(
                    f"Skipping cosine computation at step {self.engine.global_steps} as required by adjust_interval {self.params['adjust_interval']}", ranks=[0])
            else:
                # if self.skip_cos_computation:
                #     log_dist(
                #         f"Skipping cosine computation at step {self.engine.global_steps} as previous cosine value is nan",ranks=[0])
                #     self.skip_cos_computation=False
                #     # dist.barrier()
                #     return

                if self.grad_placeholder is None:
                    self.grad_placeholder = self.get_optimizer_grad(self.engine)
                start_time = time.time()

                if self.dp_group is not None:
                    self.engine.allreduce_gradients(dp_group=self.dp_group)

                # # NaN handling
                # try:
                #     assert torch.all(torch.isfinite(self.get_optimizer_grad(self.engine)))
                # except:
                #     log_dist("There are nan in SimiGrad half allreduced values",ranks=[-1])
                #     log_dist(torch.mean(torch.isfinite(self.get_optimizer_grad(self.engine)).float()),ranks=[-1])

                # dist.barrier()
                local_grad=self.get_optimizer_grad(self.engine)
                if self.grad_passing_source is not None:
                    if self.engine.global_rank == self.grad_passing_source:
                        # # NaN handling
                        # try:
                        #     assert torch.all(torch.isfinite(self.get_optimizer_grad(self.engine)))
                        # except:
                        #     log_dist("There are nan before SimiGrad half allreduced values",ranks=[-1])
                        dist.broadcast(local_grad, self.grad_passing_source, group=self.grad_passing_group)
                        # log_dist(f"Sending grads from {self.grad_passing_source}, total number {self.get_optimizer_grad(self.engine).numel()}",ranks=[-1])
                    else:
                        dist.broadcast(self.grad_placeholder, self.grad_passing_source, group=self.grad_passing_group)
                        # log_dist(f"Got grads from {self.grad_passing_source}, total number {self.grad_placeholder.numel()}",ranks=[-1])
                # dist.barrier()

                # # NaN handling
                # if self.grad_placeholder is not None and self.grad_passing_source is not None and self.engine.global_rank != self.grad_passing_source:
                #     try:
                #         assert torch.all(torch.isfinite(self.grad_placeholder))
                #     except:
                #         log_dist("There are nan in the half allreduced values other half receives",ranks=[-1])
                #         log_dist(["Source dtype",self.get_optimizer_grad(self.engine).dtype,"Receiver dtype",self.grad_placeholder.dtype,"Percentage",torch.mean(torch.isfinite(self.grad_placeholder).float()),"Number",""],ranks=[-1])

                if self.engine.mpu is None:
                    if self.engine.global_rank == 0:
                        # t1=self.get_optimizer_grad(self.engine).double()
                        # t2=self.grad_placeholder.double()
                        # finite_mask=torch.logical_and(torch.isfinite(t1),torch.isfinite(t2))
                        # self.cos_placeholder = torch.nn.functional.cosine_similarity(t1[finite_mask], t2[finite_mask], dim=0)

                        self.cos_placeholder = torch.nn.functional.cosine_similarity(local_grad.float(), self.grad_placeholder.float(), dim=0)
                else:
                    if self.grad_passing_source is not None and self.engine.global_rank != self.grad_passing_source:
                        grad_dot=torch.dot(local_grad.float(),self.grad_placeholder.float())
                        norm_first_half=torch.sum(torch.square(local_grad.float()))
                        norm_second_half=torch.sum(torch.square(self.grad_placeholder.float()))
                        # log_dist([grad_dot,norm_first_half,norm_second_half],ranks=[-1])
                        tensor_list = [torch.zeros(3).cuda().float() for _ in range(self.engine.mpu.get_model_parallel_world_size())]
                        dist.all_gather(tensor_list=tensor_list, tensor=torch.stack([grad_dot.float(),norm_first_half.float(),norm_second_half.float()]),group=self.engine.mpu.get_model_parallel_group())
                    if dist.get_rank()==0:
                        grad_dot=torch.Tensor([0]).double().cuda()
                        norm_first_half=torch.Tensor([0]).double().cuda()
                        norm_second_half=torch.Tensor([0]).double().cuda()
                        for cosine_scatter in tensor_list:
                            grad_dot+=cosine_scatter[0]
                            norm_first_half+=cosine_scatter[1]
                            norm_second_half+=cosine_scatter[2]
                        # log_dist([grad_dot,norm_first_half,norm_second_half],ranks=[-1])
                        self.cos_placeholder=grad_dot/torch.sqrt(norm_first_half)/torch.sqrt(norm_second_half)

                self.cos_placeholder=self.cos_placeholder.float()
                dist.broadcast(self.cos_placeholder, 0)
                # if not torch.all(torch.isfinite(self.cos_placeholder)):
                #     self.skip_cos_computation=True
                # log_dist(
                    # f"cosine {self.cos_placeholder.item()} computation at micro step {self.engine.micro_steps} takes {time.time()-start_time}", ranks=[0])
                self.cosine_computation_time=time.time()-start_time
        except NoAvailGradError:
            log_dist("Skipping SimiGrad cosince computation as there's no available gradient.",ranks=[0])
            return
        except Exception as e:
            raise e



    def change_micro_batch_size(self, new_micro_batch_size_per_gpu):
        self.engine._config.train_micro_batch_size_per_gpu = new_micro_batch_size_per_gpu
        log_dist(
            f"train_micro_batch_size_per_gpu:{self.engine.train_micro_batch_size_per_gpu()}", ranks=[0])

    def set_new_lr(self, new_batch_size):
        new_ratio = math.sqrt(
            new_batch_size/self.params["original_batch_size"])
        if self.params["lr_adjust_factor"] is not None:
            new_ratio = (new_ratio-1)*self.params["lr_adjust_factor"]+1
        self.params["global_lr_modifier"] = new_ratio
        log_dist(
            f"The learning rate modifier was updated to {self.params['global_lr_modifier']}", ranks=[0])

    def change_gradient_accumulation_steps(self, new_batch_size):
        log_dist(f"Target batch size is {new_batch_size}", ranks=[0])
        if self.params["smooth_factor"] is not None:
            smooth_factor = self.params["smooth_factor"]
            new_batch_size = int(self.gradient_accumulation_steps(
            )*smooth_factor+new_batch_size*(1-smooth_factor))
        if self.params["batch_size_upper_bound"] is not None:
            new_batch_size = min(
                new_batch_size, self.params["batch_size_upper_bound"])
        if self.params["batch_size_lower_bound"] is not None:  # scale up is different
            new_batch_size = max(
                new_batch_size, self.params["batch_size_lower_bound"])
        if new_batch_size > self.engine.train_batch_size():
            if self.params["max_micro_batch_size"] is not None:
                new_micro_batch_size = min(max(
                    1, new_batch_size//self.engine.dp_world_size), self.params["max_micro_batch_size"])
                self.change_micro_batch_size(new_micro_batch_size)
            new_step = max(1, new_batch_size//self.engine.dp_world_size //
                           self.engine.train_micro_batch_size_per_gpu())
            try:
                assert self.engine.dp_world_size*new_step * \
                    self.engine.train_micro_batch_size_per_gpu() > self.engine.train_batch_size()
            except:
                new_step += 1
        elif new_batch_size < self.engine.train_batch_size():
            new_step = new_batch_size//self.engine.dp_world_size//self.engine.train_micro_batch_size_per_gpu()
            try:
                assert self.engine.dp_world_size*new_step * \
                    self.engine.train_micro_batch_size_per_gpu() < self.engine.train_batch_size()
            except:
                if self.params["max_micro_batch_size"] is not None:
                    new_micro_batch_size = min(max(
                        1, new_batch_size//self.engine.dp_world_size//new_step), self.params["max_micro_batch_size"])
                    self.change_micro_batch_size(new_micro_batch_size)
                else:
                    new_step -= 1
        else:
            new_step = self.engine.gradient_accumulation_steps()
        new_step = max(1, new_step)

        self.engine._config.gradient_accumulation_steps = int(new_step)
        self.gradient_accumulation_offset = (
            self.engine.micro_steps+1) % self.engine.gradient_accumulation_steps()
        self.gradient_accumulation_start = self.engine.micro_steps + \
            self.engine.gradient_accumulation_steps()
        new_batch_size = self.engine.dp_world_size*self.engine.gradient_accumulation_steps() * \
            self.engine.train_micro_batch_size_per_gpu()
        self.engine._config.train_batch_size = int(new_batch_size)
        if self.params["disable_lr_adjust"] is not True:
            self.set_new_lr(self.engine.train_batch_size())
        # self.adapt_print(f"New gradient_accumulation_steps {self.gradient_accumulation_steps()}")
        log_dist(
            f"New batch size {self.engine.train_batch_size()}= world_size {self.engine.dp_world_size} x gradient_accumulation_steps {self.engine.gradient_accumulation_steps()} x micro_batch_size {self.engine.train_micro_batch_size_per_gpu()}", ranks=[0])
        log_dist(
            f"The offset has been updated to {self.gradient_accumulation_offset}, the next update is set to be {self.gradient_accumulation_start}, ", ranks=[0])

    def is_gradient_accumulation_boundary(self):
        # log_dist(f"{self.engine.micro_steps}>= {self.gradient_accumulation_start}: {self.engine.micro_steps >= self.gradient_accumulation_start}",ranks=[0])
        # log_dist(f"{self.engine.micro_steps} + 1 - {self.gradient_accumulation_offset} % {self.engine.gradient_accumulation_steps()}",ranks=[0])
        if self.engine.micro_steps >= self.gradient_accumulation_start:
            return (self.engine.micro_steps + 1 - self.gradient_accumulation_offset) % \
                self.engine.gradient_accumulation_steps() == 0
        else:
            return False

    def apply_lr_factor(self,lr_factor=None):
        if lr_factor is None:
            lr_factor=self.params["global_lr_modifier"]
        if lr_factor is not None:
            if hasattr(self.engine.optimizer,"optimizer"):
                for param_group in self.engine.optimizer.optimizer.param_groups:
                    param_group['lr']*=lr_factor
            else:
                for param_group in self.engine.optimizer.param_groups:
                    param_group['lr']*=lr_factor

    def remove_lr_factor(self,lr_factor=None):
        if lr_factor is None:
            lr_factor=self.params["global_lr_modifier"]
        if lr_factor is not None:
            if hasattr(self.engine.optimizer,"optimizer"):
                for param_group in self.engine.optimizer.optimizer.param_groups:
                    param_group['lr']/=lr_factor
            else:
                for param_group in self.engine.optimizer.param_groups:
                    param_group['lr']/=lr_factor


    def update_batch_size(self):
        self.params["similarity_history"].append(self.cos_placeholder.item())
        if not torch.all(torch.isfinite(self.cos_placeholder)):
            return
        if self.params["adjust_interval"] is not None and self.engine.global_steps%self.params["adjust_interval"]!=0:
            return
        else:
            if self.params["enable_adjust"]:
                log_dist(f"Current cos similarity {self.cos_placeholder} for batch size {self.engine.train_batch_size()} ", ranks=[0])
                if self.params["smoothed_similarity"]:
                    if len(self.params["similarity_history"])>20:
                        effective_cosine=np.nanmean(self.params["similarity_history"][-20:])
                    else:
                        effective_cosine=np.nanmean(self.params["similarity_history"])
                    log_dist(f"Smoothed cos similarity {effective_cosine}.", ranks=[0])
                else:
                    effective_cosine=self.cos_placeholder
                if(effective_cosine<self.params["similarity_target"]):
                    self.change_gradient_accumulation_steps(max(1,self.engine.train_batch_size()+1,int(self.engine.train_batch_size()*1.1)))
                elif(effective_cosine>self.params["similarity_target"] and self.engine.train_batch_size()>1):
                    self.change_gradient_accumulation_steps(max(1,min(self.engine.train_batch_size()-1,int(self.engine.train_batch_size()*0.9))))
                # self.adapt_print(f" to {self.train_batch_size()}"+msg+f" to {self.gradient_step_size}")
                log_dist(f"New train batch size {self.engine.train_batch_size()}", ranks=[0])

    def get_optimizer_grad(self,engine):
        # return _flatten_dense_tensors([param.grad for param_name, param in engine.module.named_parameters() if param.grad is not None])

        result = []
        # # if not engine.optimizer:
        # #     return result
        # for param_name, param in engine.module.named_parameters():
        #     if param.grad is not None:
        #         result.append(param.grad)
        # return result

        # for group in engine.basic_optimizer.param_groups:
        #     for p in group['params']:
        #         if p.grad is not None:
        #             result.append(p.grad.data)
        # return result

        if hasattr(engine.optimizer, 'fp16_groups'):
            for group in engine.optimizer.fp16_groups:
                for p in group:
                    if p.grad is not None:
                        result.append(p.grad.data)
        else:
            for group in engine.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        result.append(p.grad.data)
        if len(result)>0:
            return _flatten_dense_tensors(result)
        else:
            raise NoAvailGradError


