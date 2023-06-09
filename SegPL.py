import os
import contextlib

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import monai
from utils import prepare_batch, set_loss, Get_Scalar


class SegPL:
    def __init__(self, net_builder, num_classes, ema_m, p_cutoff, lambda_u,\
                it=0, num_eval_iter=1000, tb_log=None, logger=None):
        """
        class Fixmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes 
            ema_m: momentum of exponential moving average for eval_model
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """
        
        super(SegPL, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py
        
        self.train_model = net_builder(num_classes=num_classes) 
        self.eval_model = net_builder(num_classes=num_classes)
        self.num_eval_iter = num_eval_iter
        self.p_fn = Get_Scalar(p_cutoff) #confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        
        self.optimizer = None
        self.scheduler = None
        
        self.it = it

        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        
        for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
            
        self.eval_model.eval()
        self.supervised_loss = None
        self.unusupervised_loss = None
        self.dice_loss = monai.losses.DiceLoss(to_onehot_y=False,softmax=True,include_background=False,batch=True)
        
    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        for param_train, param_eval in zip(self.train_model.module
                                           .parameters(), self.eval_model.parameters()):
            param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1-self.ema_m))
        
        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)         
    
     
    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')    
       
    def set_supervised_loss(self, loss_name, to_onehot_y, softmax, include_background, batch):
        self.supervised_loss = set_loss(loss_name, to_onehot_y, softmax, include_background, batch)
    
    def set_unsupervised_loss(self, loss_name, to_onehot_y, softmax, include_background, batch):
        self.unsupervised_loss = set_loss(loss_name, to_onehot_y, softmax, include_background, batch)
    
    
    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
    
            
    
    def train(self, args, logger=None):
        """
        Train function of FixMatch.
        From data_loader, it inference training data, computes losses, and update the networks.
        """
        ngpus_per_node = torch.cuda.device_count()

        #lb: labeled, ulb: unlabeled
        self.train_model.train()
        
        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        
        start_batch.record()
        best_eval_dice, best_it = 1.0, 0
        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.suppress #update for python3.6
        #amp_cm = autocast if args.amp else contextlib.nullcontext #original version


        for (batch, batch_u) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break
            if self.it == args.num_train_iter:
                self.load_model(os.path.join(os.path.join(args.save_dir, args.save_name), 'model_best.pth'))
                self.it = args.num_train_iter
            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()
            
            
            
            
            x_lb, y_lb = prepare_batch(batch, args.gpu)
            x_ulb, _ = prepare_batch(batch_u, args.gpu)

            num_lb = x_lb.shape[0]            
            #weak and strong augmentations for labelled and unlabelled data

            inputs = torch.cat((x_lb, x_ulb))
            
            # inference and calculate sup/unsup losses
            with amp_cm():
                logits = self.train_model(inputs)
                logits_x_lb = logits[:num_lb]
                logits_x_ulb = logits[num_lb:]
                
                del logits
                # hyper-params for update
                p_cutoff = self.p_fn(self.it)
                
                
                # Supervised loss
                sup_loss = self.supervised_loss(logits_x_lb, y_lb)
                
                #Unsupervised losses
                pseudo_labels = (torch.nn.Softmax(dim=1)(logits_x_ulb)>p_cutoff).long().detach()
                unsup_loss = self.unsupervised_loss(logits_x_ulb, pseudo_labels)
                
                anti_pseudo_labels = (torch.nn.Softmax(dim=1)(logits_x_lb)>p_cutoff).long().detach()
                anti_unsup_loss = self.unsupervised_loss(logits_x_lb, anti_pseudo_labels)
                    
                if args.debiased:
                    total_loss = sup_loss + self.lambda_u * (unsup_loss - anti_unsup_loss)
                else:
                    total_loss = sup_loss + self.lambda_u * unsup_loss
            del logits_x_lb, logits_x_ulb
            
            if args.relu: total_loss = torch.nn.ReLU()(total_loss)
            
            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
            if self.scheduler is not None: 
                self.scheduler.step()
            self.train_model.zero_grad()
            
            torch.cuda.empty_cache()
            with torch.no_grad():
                self._eval_model_update()
            
            end_run.record()
            torch.cuda.synchronize()
            
            #tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach() 
            tb_dict['train/unsup_loss'] = unsup_loss.detach() 
            tb_dict['train/total_loss'] = total_loss.detach() 
            tb_dict['train/anti_unsup_loss'] = anti_unsup_loss.detach() 
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch)/1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run)/1000.
            
            
            if self.it % self.num_eval_iter == 0:

                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)
                
                save_path = os.path.join(args.save_dir, args.save_name)
                
                if tb_dict['eval/dice'] <= best_eval_dice:
                    best_eval_dice = tb_dict['eval/loss']
                    best_it = self.it
                
                self.print_fn(f"{self.it} iteration, USE_EMA: {hasattr(self, 'eval_model')}, {tb_dict}, BEST_EVAL_DICE: {best_eval_dice}, at {best_it} iters")
            torch.cuda.empty_cache()
            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                
                if self.it == best_it:
                    self.save_model('model_best.pth', save_path)
                if not self.tb_log is None:
                    self.tb_log.update(tb_dict, self.it)
                
            self.it +=1
            del tb_dict
            start_batch.record()
            if self.it > 2**19:
                self.num_eval_iter = 1000
        
        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_dice': best_eval_dice, 'eval/best_it': best_it})
        return eval_dict
            
            
    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        use_ema = hasattr(self, 'eval_model')
        
        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        
        total_dice = 0.0
        total_num = 0.0
    
        for batch in eval_loader:
            x, y = prepare_batch(batch, args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = eval_model(x)
            dice = self.dice_loss(logits, y)
            total_dice += dice.detach()*num_batch        
        if not use_ema:
            eval_model.train()
            
        return {'eval/dice': total_dice/total_num}

    
    
    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        torch.save({'train_model': train_model.state_dict(),
                    'eval_model': eval_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it}, save_filename)
        
        self.print_fn(f"model saved: {save_filename}")
    
    
    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        
        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if 'train_model' in key:
                    train_model.load_state_dict(checkpoint[key])
                elif 'eval_model' in key:
                    eval_model.load_state_dict(checkpoint[key])
                #elif key == 'it':
                #    self.it = checkpoint[key]
                #elif key == 'scheduler':
                #    self.scheduler.load_state_dict(checkpoint[key])
                elif key == 'it' or key == 'scheduler':
                    pass
                elif key == 'optimizer':
                    self.optimizer.load_state_dict(checkpoint[key]) 
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            else:
                self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")

if __name__ == "__main__":
    pass
