# Lecture 8: Deep Learning Foundations

* DL model Overfit
  * What is overfit? it doe NOT means your training error is lower than validation error. A well fitted model should have lower training error than validation error. You should see your validation error got worse if you overfit 
  * 5 steps to avoid overfit
    1. more data 
    2. data augmentation 
    3. generalizable architectures 
    4. regularization (drop out, weight decay) 
    5. reduce architecture complexity
* Reading papers 
  * Don't be intimidated by the greek letters. (e.g. Adam paper has complex equations but can be implement using excel) 
  * Get comfortable of reading complex papers. Find blog posts describing the paper to help understanding
  * Learn to pronounce Greek letter
* Pytorch
  * Try to always use torch tensor (take advantage of gpu)
  * tensor `.view()` function creates / reshape a new tensor with different shape, e.g. if the image data come as in 1D, we convert to 2D for plotting.
* Broadcasting
  * matrix does not need to be in the same rank, e.g. in vision, we can normalize channels using broadcasting 
  * Normalized training set, we need to apply the same  training mean and std to the validation set
* Initialization is critical 
  * Pytorch use He normalize the tensor to unitform normal. 
  * Don't forgot Relu require further work => that is why He initialization made a break through to correct 
  * Common initialization
    * Xavier 
    * Kaimin He 
    * Pytorch `std = sqrt(2/(1+a^2) * fan_in)` 
    * `init.kaiming_normal_(w1, mode='fan_out')` => pytorch does transpose due to legacy issue that is why it uses fan_out

# Lecture 9: Model Training Loop (Data loader/bunch, optimizer, callbacks)
* Cross entropy
  * softmax on prediction => log(softmax on prediction)  => negative log likelyhood on (log(softmax on prediciton), target) is the mean of -sum(tx * log(pred(x))  i.e. cross entropy loss
  * In Pytorch `F.log_softmax` and `F.nll_loss` are combined in one optimized function, `F.cross_entropy`
  * there are steps to optimize how to calculate log softmax and how we get target actual for NLL function
  * Use **logsumexp** to get more mathmatical stable calculate.
* Python:
  *  `__setattr__` function allow us to set attribute from variable defined in the `__init__()`
* Pytorch: 
  * classes for define model: `nn.Module` `nn.Module_list`
    * `nn.Module` is the based class for all neural network modules.
  * module for define optimizer is `optim` e.g. `optim.SGD`
* Jeremy don't set random seed when development models (it is different from reproducible science). When development models, you want to see sth unexpected (variation)
* Why zero out gradients
  * if not, the gradients will add to existing gradients, we have gradients at many placed when we run backwards, we don't want to zero out there. Thus, we want to zero out after the calculation of gradient
  * We also don't want to remove the possibility of not zero the gradient in the optimizer function.

* Training loop
  * Training Loop Steps:
    1. Calculate prediction
    2. calculate loss
    3. back propagate
    4. subtract learning rate * gradient (for all weights)
    5. zero the gradients
  * After defined dataset, dataloader class, defined get_model function

    ```python
    class Dataset():
        def __init__(self, x, y): self.x,self.y = x,y
        def __len__(self): return len(self.x)
        def __getitem__(self, i): return self.x[i],self.y[i]
    train_ds,valid_ds = Dataset(x_train, y_train),Dataset(x_valid, y_valid)

    class DataLoader():
        def __init__(self, ds, bs): self.ds,self.bs = ds,bs
        def __iter__(self):
            for i in range(0, len(self.ds), self.bs): yield self.ds[i:i+self.bs]

    train_dl = DataLoader(train_ds, bs)
    valid_dl = DataLoader(valid_ds, bs)

    def get_model():
        model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))
        return model, optim.SGD(model.parameters(), lr=lr)
    model,opt = get_model()
    loss_func = F.cross_entropy

    def fit():
        for epoch in range(epochs):
            for xb,yb in train_dl: # iterate from a dataloader (batch)
                pred = model(xb) # calculate predictions
                loss = loss_func(pred, yb) # calculate loss
                loss.backward() # Calculate gradients
                opt.step() # Update with the learning rate
                opt.zero_grad() # Reset gradients
    ```

* DataLoader
  * Training set needs to be in random order but validation set should not be randomized
  * DL datasets often in the order of predict variable (e.g. image in the order of class)
  * Custom random sampler class and dataLoader
    ```python
    class Sampler():
    def __init__(self, ds, bs, shuffle=False):
        self.n,self.bs,self.shuffle = len(ds),bs,shuffle
        
    def __iter__(self):
        self.idxs = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for i in range(0, self.n, self.bs): yield self.idxs[i:i+self.bs]
    
    def collate(b):
        xs,ys = zip(*b)
        return torch.stack(xs),torch.stack(ys)

    class DataLoader():
        def __init__(self, ds, sampler, collate_fn=collate):
            self.ds,self.sampler,self.collate_fn = ds,sampler,collate_fn
            
        def __iter__(self):
            for s in self.sampler: yield self.collate_fn([self.ds[i] for i in s])
    ```

      * Using `__iter__` to enable generator capability to yield samples
      * `shuffle` parameter allow ranodm for train, no random for validation
      * `collate` function helps to put samples together such as stacking, padding, etc
    
  * Pytorch DataLoader
    ```python
    from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

    # using custom collate function
    train_dl = DataLoader(train_ds, bs, sampler=RandomSampler(train_ds), collate_fn=collate)
    valid_dl = DataLoader(valid_ds, bs, sampler=SequentialSampler(valid_ds), collate_fn=collate)

    # or just use default
    train_dl = DataLoader(train_ds, bs, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, bs, shuffle=False)
    ```
* Training loop + Validation => the full fit function

    ```python
    def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
        for epoch in range(epochs):
            # Handle batchnorm / dropout
            model.train()
    #         print(model.training)
            for xb,yb in train_dl:
                loss = loss_func(model(xb), yb)
                loss.backward()
                opt.step()
                opt.zero_grad()

            model.eval()
    #         print(model.training)
            with torch.no_grad():
                tot_loss,tot_acc = 0.,0.
                for xb,yb in valid_dl:
                    pred = model(xb)
                    tot_loss += loss_func(pred, yb)
                    tot_acc  += accuracy (pred,yb)
            nv = len(valid_dl)
            print(epoch, tot_loss/nv, tot_acc/nv)
        return tot_loss/nv, tot_acc/nv

    def get_dls(train_ds, valid_ds, bs, **kwargs):
        return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
                DataLoader(valid_ds, batch_size=bs*2, **kwargs))

    train_dl,valid_dl = get_dls(train_ds, valid_ds, bs)
    model,opt = get_model()
    loss,acc = fit(5, model, loss_func, opt, train_dl, valid_dl)
    ```
    * we have a similar loop but no back prop and step optimizer
    * Pytorch `model.train()` and `model.eval()` set an internal attributes `model.training` to `True` or `False`, we need this flag because some of layers such as `nn.BatchNorm` and `nn.Dropout` have different behaviors in training or validation
    * The final loss and accuracy is not totally correct if have different last mini-batch size but most existing framework does this way. Ideally, we want to use the weighted average
* Why we zero out gradients
  * gradient tell us the direction of optimization (to global maximum/minimum), there are gradient from any places. To avoid moving toward a wrong direction, we want to have zero to start before perform back-prop
  * If not, the gradients will add to existing gradients (pytorch), we have gradients at many placed when we run backwards, we don't want to zero out there. Thus, we want to zero out after the calculation of gradient
  * We also don't want to remove the possibility of not zero the gradient in the optimizer function.

* DataBunch and Learner
  * They are wrapper (abostraction) class that help us organize our code so we don't have to pass many items to the fit function
  * Items used by `fit()`: `epoch`, `model`, `loss_func`, `opt`, `train_dl`, `valid_dl`
  * Use DataBunch class to combine train and validation DataLoader and also take care of final acitivation layer (num of classes)
    ```python
    class DataBunch():
        def __init__(self, train_dl, valid_dl, c=None):
            self.train_dl,self.valid_dl,self.c = train_dl,valid_dl,c
            
        @property
        def train_ds(self): return self.train_dl.dataset
            
        @property
        def valid_ds(self): return self.valid_dl.dataset

    data = DataBunch(*get_dls(train_ds, valid_ds, bs), c)
    ```
  * User Learner function to wrap `DataBunch` along with `model`, `loss_func`, and `opt`
    ```python
    def get_model(data, lr=0.5, nh=50):
        m = data.train_ds.x.shape[1]
        model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,data.c))
        return model, optim.SGD(model.parameters(), lr=lr)

    class Learner():
        def __init__(self, model, opt, loss_func, data):
            self.model,self.opt,self.loss_func,self.data = model,opt,loss_func,data
    
    learn = Learner(*get_model(data), loss_func, data)
    ```

  * Optimized fit function:
    ```python
    def fit(epochs, learn):
        for epoch in range(epochs)
            learn.model.train()
            for xb,yb in learn.data.train_dl:
                loss = learn.loss_func(learn.model(xb), yb)
                loss.backward()
                learn.opt.step()
                learn.opt.zero_grad()

            learn.model.eval()
            with torch.no_grad():
                tot_loss,tot_acc = 0.,0.
                for xb,yb in learn.data.valid_dl:
                    pred = learn.model(xb)
                    tot_loss += learn.loss_func(pred, yb)
                    tot_acc  += accuracy (pred,yb)
            nv = len(learn.data.valid_dl)
            print(epoch, tot_loss/nv, tot_acc/nv)
        return tot_loss/nv, tot_acc/nv
    
    loss, acc = fit(1, learn)
    ```

* Callbacks
  * What is a callback?: A callback is a function that is passed as an argument to other function. This other function is expected to call this callback function in its definition. The point at which other function calls our callback function depends on the requirement and nature of other function. Callback Functions are generally used with asynchronous functions. In deep learning, we use callback to execute actions before or after certain steps of training or validation

  * Example (Brute force implementation)
    ```python
    def one_batch(xb, yb, cb):
        if not cb.begin_batch(xb,yb): return
        loss = cb.learn.loss_func(cb.learn.model(xb), yb)
        if not cb.after_loss(loss): return
        loss.backward()
        if cb.after_backward(): cb.learn.opt.step()
        if cb.after_step(): cb.learn.opt.zero_grad()

    def all_batches(dl, cb):
        for xb,yb in dl:
            one_batch(xb, yb, cb)
            if cb.do_stop(): return

    # Using call back by passing a callback handler object to fit function
    def fit(epochs, learn, cb):
        if not cb.begin_fit(learn): return
        for epoch in range(epochs):
            if not cb.begin_epoch(epoch): continue
            all_batches(learn.data.train_dl, cb)
            
            if cb.begin_validate():
                with torch.no_grad(): all_batches(learn.data.valid_dl, cb)
            if cb.do_stop() or not cb.after_epoch(): break
        cb.after_fit()
    
    class Callback():
        def begin_fit(self, learn):
            self.learn = learn
            return True
        def after_fit(self): return True
        def begin_epoch(self, epoch):
            self.epoch=epoch
            return True
        def begin_validate(self): return True
        def after_epoch(self): return True
        def begin_batch(self, xb, yb):
            self.xb,self.yb = xb,yb
            return True
        def after_loss(self, loss):
            self.loss = loss
            return True
        def after_backward(self): return True
        def after_step(self): return True

    class CallbackHandler():
        def __init__(self,cbs=None):
            self.cbs = cbs if cbs else []

        def begin_fit(self, learn):
            self.learn,self.in_train = learn,True
            learn.stop = False
            res = True
            for cb in self.cbs: res = res and cb.begin_fit(learn)
            return res

        def after_fit(self):
            res = not self.in_train
            for cb in self.cbs: res = res and cb.after_fit()
            return res
        
        def begin_epoch(self, epoch):
            self.learn.model.train()
            self.in_train=True
            res = True
            for cb in self.cbs: res = res and cb.begin_epoch(epoch)
            return res

        def begin_validate(self):
            self.learn.model.eval()
            self.in_train=False
            res = True
            for cb in self.cbs: res = res and cb.begin_validate()
            return res

        def after_epoch(self):
            res = True
            for cb in self.cbs: res = res and cb.after_epoch()
            return res
        
        def begin_batch(self, xb, yb):
            res = True
            for cb in self.cbs: res = res and cb.begin_batch(xb, yb)
            return res

        def after_loss(self, loss):
            res = self.in_train
            for cb in self.cbs: res = res and cb.after_loss(loss)
            return res

        def after_backward(self):
            res = True
            for cb in self.cbs: res = res and cb.after_backward()
            return res

        def after_step(self):
            res = True
            for cb in self.cbs: res = res and cb.after_step()
            return res
        
        def do_stop(self):
            try:     return self.learn.stop
            finally: self.learn.stop = False
    
    # Define a callback by inherit callback class
    class TestCallback(Callback):
        def begin_fit(self,learn):
            super().begin_fit(learn)
            self.n_iters = 0
            return True
            
        def after_step(self):
            self.n_iters += 1
            print(self.n_iters)
            if self.n_iters>=10: self.learn.stop = True
            return True
    
    # Pass callback object to callback handler then to fit function
    fit(1, learn, cb=CallbackHandler([TestCallback()]))
    ```

    * This implementation have a lot of repeated code snippet and is not the best software engineering practice
    * Looks complex and not flexible

  * A better Implementation but not obvious
    ```python
    import re

    _camel_re1 = re.compile('(.)([A-Z][a-z]+)')
    _camel_re2 = re.compile('([a-z0-9])([A-Z])')
    def camel2snake(name):
        s1 = re.sub(_camel_re1, r'\1_\2', name)
        return re.sub(_camel_re2, r'\1_\2', s1).lower()

    class Callback():
        _order=0
        def set_runner(self, run): self.run=run
        def __getattr__(self, k): return getattr(self.run, k)
        @property
        def name(self):
            name = re.sub(r'Callback$', '', self.__class__.__name__)
            return camel2snake(name or 'callback')
    
    from typing import *

    def listify(o):
        if o is None: return []
        if isinstance(o, list): return o
        if isinstance(o, str): return [o]
        if isinstance(o, Iterable): return list(o)
        return [o]
    
    class Runner():
        def __init__(self, cbs=None, cb_funcs=None):
            cbs = listify(cbs)
            for cbf in listify(cb_funcs):
                cb = cbf()
                setattr(self, cb.name, cb)
                cbs.append(cb)
            self.stop,self.cbs = False,[TrainEvalCallback()]+cbs

        @property
        def opt(self):       return self.learn.opt
        @property
        def model(self):     return self.learn.model
        @property
        def loss_func(self): return self.learn.loss_func
        @property
        def data(self):      return self.learn.data

        def one_batch(self, xb, yb):
            self.xb,self.yb = xb,yb
            if self('begin_batch'): return
            self.pred = self.model(self.xb)
            if self('after_pred'): return
            self.loss = self.loss_func(self.pred, self.yb)
            if self('after_loss') or not self.in_train: return
            self.loss.backward()
            if self('after_backward'): return
            self.opt.step()
            if self('after_step'): return
            self.opt.zero_grad()

        def all_batches(self, dl):
            self.iters = len(dl)
            for xb,yb in dl:
                if self.stop: break
                self.one_batch(xb, yb)
                self('after_batch')
            self.stop=False

        def fit(self, epochs, learn):
            self.epochs,self.learn = epochs,learn

            try:
                for cb in self.cbs: cb.set_runner(self)
                if self('begin_fit'): return
                for epoch in range(epochs):
                    self.epoch = epoch
                    if not self('begin_epoch'): self.all_batches(self.data.train_dl)

                    with torch.no_grad(): 
                        if not self('begin_validate'): self.all_batches(self.data.valid_dl)
                    if self('after_epoch'): break
                
            finally:
                self('after_fit')
                self.learn = None

        def __call__(self, cb_name):
            for cb in sorted(self.cbs, key=lambda x: x._order):
                f = getattr(cb, cb_name, None)
                if f and f(): return True
            return False
    
    # Define callback classes
    class TrainEvalCallback(Callback):
        def begin_fit(self):
            self.run.n_epochs=0.
            self.run.n_iter=0

        def after_batch(self):
            if not self.in_train: return
            self.run.n_epochs += 1./self.iters
            self.run.n_iter   += 1
            
        def begin_epoch(self):
            self.run.n_epochs=self.epoch
            self.model.train()
            self.run.in_train=True

        def begin_validate(self):
            self.model.eval()
            self.run.in_train=False
    
    class AvgStats():
        def __init__(self, metrics, in_train): self.metrics,self.in_train = listify(metrics),in_train
        
        def reset(self):
            self.tot_loss,self.count = 0.,0
            self.tot_mets = [0.] * len(self.metrics)
            
        @property
        def all_stats(self): return [self.tot_loss.item()] + self.tot_mets
        @property
        def avg_stats(self): return [o/self.count for o in self.all_stats]
        
        def __repr__(self):
            if not self.count: return ""
            return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

        def accumulate(self, run):
            bn = run.xb.shape[0]
            self.tot_loss += run.loss * bn
            self.count += bn
            for i,m in enumerate(self.metrics):
                self.tot_mets[i] += m(run.pred, run.yb) * bn

    class AvgStatsCallback(Callback):
        def __init__(self, metrics):
            self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)
            
        def begin_epoch(self):
            self.train_stats.reset()
            self.valid_stats.reset()
            
        def after_loss(self):
            stats = self.train_stats if self.in_train else self.valid_stats
            with torch.no_grad(): stats.accumulate(self.run)
        
        def after_epoch(self):
            print(self.train_stats)
            print(self.valid_stats)
    
    class TestCallback(Callback):
        _order=1
        def after_step(self):
            if self.train_eval.n_iters>=10: return True
    ```
     * Callback class is very simple
       * It expect a runner object, and get attributes from the runner object and a read-only name property
       * It has an `_order` attribute that allow us refer order of class inheritance such as in `TestCallBack`
        * `__getattr__` inside callback class is a way to let callback to get the attributes from the runner (if it cannot find within callbacks makes callback class simplier, we will get them from runner), we may needs those attrutes in callbacks. This enable us to use `self.` syntax to get attributes from runner. (In Python, `__getattr__` only called when the class object cannot find an attribute)
          * This is a software design pattern that we can delegate to other object if one object calls another object.
    * The Runner class does the job of wrapping all the object under a class and orchastrate the train, valid, and callbacks.
      * `self(callback_name)` is essentially `self.__call__(callback_name)`
      * its `__call__()` function orchastrates the callbacks

    * `__repr__` Returns a string as a representation of the object. Ideally, the representation should be information-rich and could be used to recreate an object with the same value.
    * `partial` allow us to fix a certain number of arguments of a function and generate a new function.

* Annealing
  * The first batch often determine how well a model can be trained (get super convergence) so it is important to ensure the model train well in the first batch
  * We define two new callbacks: the Recorder to save track of the loss and our scheduled learning rate, and a ParamScheduler that can schedule any hyperparameter as long as it's registered in the state_dict of the optimizer.
    ```python
    class Recorder(Callback):
        def begin_fit(self): self.lrs,self.losses = [],[]

        def after_batch(self):
            if not self.in_train: return
            self.lrs.append(self.opt.param_groups[-1]['lr'])
            self.losses.append(self.loss.detach().cpu())        

        def plot_lr  (self): plt.plot(self.lrs)
        def plot_loss(self): plt.plot(self.losses)

    class ParamScheduler(Callback):
        _order=1
        def __init__(self, pname, sched_func): self.pname,self.sched_func = pname,sched_func

        def set_param(self):
            for pg in self.opt.param_groups:
                pg[self.pname] = self.sched_func(self.n_epochs/self.epochs)
                
        def begin_batch(self): 
            if self.in_train: self.set_param()
    
    def annealer(f):
        def _inner(start, end): return partial(f, start, end)
        return _inner

    @annealer
    def sched_lin(start, end, pos): return start + pos*(end-start)

    @annealer
    def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2

    @annealer
    def sched_no(start, end, pos):  return start

    @annealer
    def sched_exp(start, end, pos): return start * (end/start) ** pos

    def cos_1cycle_anneal(start, high, end):
        return [sched_cos(start, high), sched_cos(high, end)]

    def combine_scheds(pcts, scheds):
        assert sum(pcts) == 1.
        pcts = tensor([0] + listify(pcts))
        assert torch.all(pcts >= 0)
        pcts = torch.cumsum(pcts, 0)
        def _inner(pos):
            idx = (pos >= pcts).nonzero().max()
            if idx == 2: idx = 1
            actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
            return scheds[idx](actual_pos)
        return _inner

    sched = combine_scheds([0.3, 0.7], [sched_cos(0.3, 0.6), sched_cos(0.6, 0.2)]) 
    ```
    * `ParamScheduler` 's `set_param` function will use `sched_func` to calculate what learning we need to pass to parameter groups. `sched_func` takes a pos in the training epochs and return a proper learning rate based on the type of learning rate schedule
    * We create a annealer decorater based on partial to enable different learning rate scheduler 
    * `ParamScheduler` is set to be `_order=1`
    * `combine_scheds` allow us to stitch learning rate schedule based on % of epochs

* Put everything together

    ```python
    cbfs = [Recorder,
            partial(AvgStatsCallback,accuracy),
            partial(ParamScheduler, 'lr', sched)]
    
    learn = create_learner(get_model_func(0.3), loss_func, data)
    run = Runner(cb_funcs=cbfs)
    run.fit(3, learn)
    run.recorder.plot_lr()
    ```

# Lecture 10: Dive inside mode (more callbacks, hooks, batch norm)

* Callbacks in software concepts
  * Concept: A mechanism to pass a function as an argument to enable some actions from that function
  * We can leverage `lambda` and `partials` to construct customizable callbacks
    * `lambda` is a way to create a function on the fly but it is hard to work with multiple arguments, we can create an inner function in a function such as:
        ```python
        def make_show_progress(exclamation):
            _inner = lambda epoch: print(f"{exclamation}! We've finished epoch {epoch}!")
            return _inner
        
        # or without lambda (closure function)
        def make_show_progress(exclamation):
            # Leading "_" is generally understood to be "private"
            def _inner(epoch): print(f"{exclamation}! We've finished epoch {epoch}!")
            return _inner
        
        slow_calculation(make_show_progress("Nice!"))
        ```
    * `partial` help to create a new function with less parameters and perform the same task
        ```python
        def show_progress(exclamation, epoch):
            print(f"{exclamation}! We've finished epoch {epoch}!")

        slow_calculation(partial(show_progress, "OK I guess"))
        ```
  * Use class
    ```python
    class ProgressShowingCallback():
        def __init__(self, exclamation="Awesome"): self.exclamation = exclamation
        def __call__(self, epoch): print(f"{self.exclamation}! We've finished epoch {epoch}!")
    
    cb = ProgressShowingCallback("Just super")
    show_calculation(cb)
    ```

* Anything that looks like `__this__` is, in some way, *special*. Python, or some library, can define some functions that they will call at certain documented times. For instance, when your class is setting up a new object, python will call `__init__`. These are defined as part of the python [data model](https://docs.python.org/3/reference/datamodel.html#object.__init__).  For instance, if python sees `+`, then it will call the special method `__add__`. If you try to display an object in Jupyter (or lots of other places in Python) it will call `__repr__`. 

  * Special methods we should probably know about (see data model link above) are:

    - `__getitem__`
    - `__getattr__`
    - `__setattr__`
    - `__del__`
    - `__init__`
    - `__new__`
    - `__enter__`
    - `__exit__`
    - `__len__`
    - `__repr__`
    - `__str__`

* Variance concept
  * Standard deviation
  ```python
  t = torch.tensor([1.,2.,4.,18])
  m = t.mean()
  (t-m).pow(2).mean().sqrt()
  ```
  * Mean Absolution deviation
  ```python
  (t-m).abs().mean()
  ```
  * standard deviation is more sensitive to outliers due to power of 2. For deep learning, mean absolution deviation is more robust because we often have outlies.

  * This expression is much easier to work with: you only have to track two things: the sum of the data, and the sum of squares of the data.
    * $$\operatorname{E}\left[X^2 \right] - \operatorname{E}[X]^2$$
    ```python
    (t*t).mean() - (m*m)
    ```

  * Covariance and correlation
    * Covariance
      * $\operatorname{cov}(X,Y) = \operatorname{E}{\big[(X - \operatorname{E}[X])(Y - \operatorname{E}[Y])\big]}$ but similarly it is easier to use $\operatorname{E}\left[X Y\right] - \operatorname{E}\left[X\right] \operatorname{E}\left[Y\right]$ instead
        ```python
        cov = (t*v).mean() - t.mean()*v.mean()
        ```
    * Correlation
      * Pearson correlation coefficient: $\rho_{X,Y}= \frac{\operatorname{cov}(X,Y)}{\sigma_X \sigma_Y}$
        ```python
        corr = cov / (t.std() * v.std())
        ```
* Softmax is not always a good metric for image classification because it always needs to pick out a winner even if a image may not have any target object. Softmax is a good metrics for NLP since we always have a word

* Callback with further enhanced implementation using exceptions and new functionalities such as **Learning Rate Finder**
    * We want to use exception mechanism to be able to control the training flow
      * Inherit Exception class to get all its behavior with different class name
    * We can move the checking of callback function to callback class from runner class

    ```python
    class Callback():
        _order=0
        def set_runner(self, run): self.run=run
        def __getattr__(self, k): return getattr(self.run, k)
        
        @property
        def name(self):
            name = re.sub(r'Callback$', '', self.__class__.__name__)
            return camel2snake(name or 'callback')
        
        def __call__(self, cb_name):
            f = getattr(self, cb_name, None)
            if f and f(): return True
            return False

    class TrainEvalCallback(Callback):
        def begin_fit(self):
            self.run.n_epochs=0.
            self.run.n_iter=0
        
        def after_batch(self):
            if not self.in_train: return
            self.run.n_epochs += 1./self.iters
            self.run.n_iter   += 1
            
        def begin_epoch(self):
            self.run.n_epochs=self.epoch
            self.model.train()
            self.run.in_train=True

        def begin_validate(self):
            self.model.eval()
            self.run.in_train=False

    class CancelTrainException(Exception): pass
    class CancelEpochException(Exception): pass
    class CancelBatchException(Exception): pass
    class Runner():
        def __init__(self, cbs=None, cb_funcs=None):
            self.in_train = False
            cbs = listify(cbs)
            for cbf in listify(cb_funcs):
                cb = cbf()
                setattr(self, cb.name, cb)
                cbs.append(cb)
            self.stop,self.cbs = False,[TrainEvalCallback()]+cbs

        @property
        def opt(self):       return self.learn.opt
        @property
        def model(self):     return self.learn.model
        @property
        def loss_func(self): return self.learn.loss_func
        @property
        def data(self):      return self.learn.data

        def one_batch(self, xb, yb):
            try:
                self.xb,self.yb = xb,yb
                self('begin_batch')
                self.pred = self.model(self.xb)
                self('after_pred')
                self.loss = self.loss_func(self.pred, self.yb)
                self('after_loss')
                if not self.in_train: return
                self.loss.backward()
                self('after_backward')
                self.opt.step()
                self('after_step')
                self.opt.zero_grad()
            except CancelBatchException: self('after_cancel_batch')
            finally: self('after_batch')

        def all_batches(self, dl):
            self.iters = len(dl)
            try:
                for xb,yb in dl: self.one_batch(xb, yb)
            except CancelEpochException: self('after_cancel_epoch')

        def fit(self, epochs, learn):
            self.epochs,self.learn,self.loss = epochs,learn,tensor(0.)

            try:
                for cb in self.cbs: cb.set_runner(self)
                self('begin_fit')
                for epoch in range(epochs):
                    self.epoch = epoch
                    if not self('begin_epoch'): self.all_batches(self.data.train_dl)

                    with torch.no_grad(): 
                        if not self('begin_validate'): self.all_batches(self.data.valid_dl)
                    self('after_epoch')
                
            # call-back function using CancelTrainException
            except CancelTrainException: self('after_cancel_train')
            finally:
                self('after_fit')
                self.learn = None

        def __call__(self, cb_name):
            res = False
            for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) or res
            return res
        
    class TestCallback(Callback):
        _order=1
        def after_step(self):
            print(self.n_iter)
            if self.n_iter>=10: raise CancelTrainException()

    run = Runner(cb_funcs=TestCallback)
    run.fit(3, learn)
    ```

    Define more useful callbacks
    ```python
    class AvgStatsCallback(Callback):
        def __init__(self, metrics):
            self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)
            
        def begin_epoch(self):
            self.train_stats.reset()
            self.valid_stats.reset()
            
        def after_loss(self):
            stats = self.train_stats if self.in_train else self.valid_stats
            with torch.no_grad(): stats.accumulate(self.run)
        
        def after_epoch(self):
            print(self.train_stats)
            print(self.valid_stats)
        
    class Recorder(Callback):
        def begin_fit(self):
            self.lrs = [[] for _ in self.opt.param_groups]
            self.losses = []

        def after_batch(self):
            if not self.in_train: return
            for pg,lr in zip(self.opt.param_groups,self.lrs): lr.append(pg['lr'])
            self.losses.append(self.loss.detach().cpu())        

        def plot_lr  (self, pgid=-1): plt.plot(self.lrs[pgid])
        def plot_loss(self, skip_last=0): plt.plot(self.losses[:len(self.losses)-skip_last])
            
        def plot(self, skip_last=0, pgid=-1):
            losses = [o.item() for o in self.losses]
            lrs    = self.lrs[pgid]
            n = len(losses)-skip_last
            plt.xscale('log')
            plt.plot(lrs[:n], losses[:n])

    class ParamScheduler(Callback):
        _order=1
        def __init__(self, pname, sched_funcs): self.pname,self.sched_funcs = pname,sched_funcs
            
        def begin_fit(self):
            if not isinstance(self.sched_funcs, (list,tuple)):
                self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

        def set_param(self):
            assert len(self.opt.param_groups)==len(self.sched_funcs)
            for pg,f in zip(self.opt.param_groups,self.sched_funcs):
                pg[self.pname] = f(self.n_epochs/self.epochs)
                
        def begin_batch(self): 
            if self.in_train: self.set_param()
    
    class LR_Find(Callback):
        _order=1
        def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
            self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr
            self.best_loss = 1e9
            
        def begin_batch(self): 
            if not self.in_train: return
            pos = self.n_iter/self.max_iter
            lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
            for pg in self.opt.param_groups: pg['lr'] = lr
                
        def after_step(self):
            if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:
                raise CancelTrainException()
            if self.loss < self.best_loss: self.best_loss = self.loss
    
    learn = create_learner(get_model, loss_func, data)
    run = Runner(cb_funcs=[LR_Find, Recorder])
    run.fit(2. learn)
    ```

* Cuda, CNN

  * CNN model using a `Lambda` class
    ```python
    class Lambda(nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func

        def forward(self, x): return self.func(x)

    def flatten(x):      return x.view(x.shape[0], -1)
    def get_cnn_model(data):
        return nn.Sequential(
            Lambda(mnist_resize),
            nn.Conv2d( 1, 8, 5, padding=2,stride=2), nn.ReLU(), #14
            nn.Conv2d( 8,16, 3, padding=1,stride=2), nn.ReLU(), # 7
            nn.Conv2d(16,32, 3, padding=1,stride=2), nn.ReLU(), # 4
            nn.Conv2d(32,32, 3, padding=1,stride=2), nn.ReLU(), # 2
            nn.AdaptiveAvgPool2d(1),
            Lambda(flatten),
            nn.Linear(32,data.c)
        )

    model = get_cnn_model(data)
    cbfs = [Recorder, partial(AvgStatsCallback,accuracy)]
    opt = optim.SGD(model.parameters(), lr=0.4)
    learn = Learner(model, opt, loss_func, data)
    run = Runner(cb_funcs=cbfs)
    run.fit(1, learn)
    ```

  * Define CUDA callback and run on GPU

  ```python
  device = torch.device('cuda',0)
  class CudaCallback(Callback):
    def __init__(self,device): self.device=device
    def begin_fit(self): self.model.to(self.device)
    def begin_batch(self): self.run.xb,self.run.yb = self.xb.to(self.device),self.yb.to(self.device)

  # Alternatively
  torch.cuda.set_device(device)
  class CudaCallback(Callback):
    def begin_fit(self): self.model.cuda()
    def begin_batch(self): self.run.xb,self.run.yb = self.xb.cuda(),self.yb.cuda()
  cbfs.append(CudaCallback)
  opt = optim.SGD(model.parameters(), lr=0.4)
  learn = Learner(model, opt, loss_func, data)
  run = Runner(cb_funcs=cbfs)
  run.fit(1, learn)
  ```

  * Improve how we define a CNN model
    * Can pass layer size as a list
    * Use callback to perform input size transformation

    ```python
    # define a single layer of cnn2d + relu and wrap in nn.Sequential
    def conv2d(ni, nf, ks=3, stride=2):
        return nn.Sequential(
            nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride), nn.ReLU())

    # Callback to perform input size conversion
    class BatchTransformXCallback(Callback):
        _order=2
        def __init__(self, tfm): self.tfm = tfm
        def begin_batch(self): self.run.xb = self.tfm(self.xb)

    # a partial function that performs size conversion
    def view_tfm(*size):
        def _inner(x): return x.view(*((-1,)+size))
        return _inner

    # create mnist_view function using the partial function view_tfm
    mnist_view = view_tfm(1,28,28)

    # Add callback to callbacks, notice(we used partial when define the callback func)
    cbfs.append(partial(BatchTransformXCallback, mnist_view))

    # A generic create CNN model function
    # Define filter size of each layer
    nfs = [8,16,32,32]

    # Create layers
    def get_cnn_layers(data, nfs):
        nfs = [1] + nfs
        return [
            # We start with kernel size 5 because from 3x3x1=9 to 8, we would not gain info
            conv2d(nfs[i], nfs[i+1], 5 if i==0 else 3) 
            for i in range(len(nfs)-1)
        ] + [nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(nfs[-1], data.c)]

    # Create model
    def get_cnn_model(data, nfs): return nn.Sequential(*get_cnn_layers(data, nfs))

    # Create a runner (all steps)
    def get_runner(model, data, lr=0.6, cbs=None, opt_func=None, loss_func = F.cross_entropy):
        if opt_func is None: opt_func = optim.SGD
        opt = opt_func(model.parameters(), lr=lr)
        learn = Learner(model, opt, loss_func, data)
        return learn, Runner(cb_funcs=listify(cbs))

    model = get_cnn_model(data, nfs)
    learn,run = get_runner(model, data, lr=0.4, cbs=cbfs)
    run.fit(3, learn)
    ```

* Hooks
  * Pytorch hooks in `nn.Module.register_forward_hook(hook)`, `nn.Module.register_backward_hook(hook)`
    hook definition 
    ```python
    hook(module, input, output) -> None or modified output
    ```
    Example
    ```python
    model = get_cnn_model(data, nfs)
    learn,run = get_runner(model, data, lr=0.5, cbs=cbfs)
    act_means = [[] for _ in model]
    act_stds  = [[] for _ in model]

    def append_stats(i, mod, inp, outp):
        act_means[i].append(outp.data.mean())
        act_stds [i].append(outp.data.std())

    # define forward hooks to all layers 
    for i,m in enumerate(model): m.register_forward_hook(partial(append_stats, i))
    ```

  * Custom hook
    * hook class

    Define a `remove()` and `__del__()` to help perform memory garbage clean

        ```python
        def children(m): return list(m.children())

        class Hook():
            def __init__(self, m, f): self.hook = m.register_forward_hook(partial(f, self))
            def remove(self): self.hook.remove()
            def __del__(self): self.remove()

        def append_stats(hook, mod, inp, outp):
            if not hasattr(hook,'stats'): hook.stats = ([],[])
            means,stds = hook.stats
            means.append(outp.data.mean())
            stds .append(outp.data.std())
        
        model = get_cnn_model(data, nfs)
        learn,run = get_runner(model, data, lr=0.5, cbs=cbfs)
        hooks = [Hook(l, append_stats) for l in children(model[:4])]
        run.fit(1, learn)
        ```
    
    * hooks class

      * Use a custom list container class to manage hooks
      * Enable context manager to hooks so we can turn off and clean up after usage

        ```python
        class ListContainer():
            def __init__(self, items): self.items = listify(items)
            def __getitem__(self, idx):
                if isinstance(idx, (int,slice)): return self.items[idx]
                if isinstance(idx[0],bool):
                    assert len(idx)==len(self) # bool mask
                    return [o for m,o in zip(idx,self.items) if m]
                return [self.items[i] for i in idx]
            def __len__(self): return len(self.items)
            def __iter__(self): return iter(self.items)
            def __setitem__(self, i, o): self.items[i] = o
            def __delitem__(self, i): del(self.items[i])
            def __repr__(self):
                res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
                if len(self)>10: res = res[:-1]+ '...]'
                return res
        
        from torch.nn import init

        class Hooks(ListContainer):
            def __init__(self, ms, f): super().__init__([Hook(m, f) for m in ms])
            def __enter__(self, *args): return self
            def __exit__ (self, *args): self.remove()
            def __del__(self): self.remove()

            def __delitem__(self, i):
                self[i].remove()
                super().__delitem__(i)
                
            def remove(self):
                for h in self: h.remove()
        
        model = get_cnn_model(data, nfs).cuda()
        learn,run = get_runner(model, data, lr=0.9, cbs=cbfs)
        for l in model:
            if isinstance(l, nn.Sequential):
                init.kaiming_normal_(l[0].weight)
                l[0].bias.data.zero_()

        with Hooks(model, append_stats) as hooks:
            run.fit(2, learn)
            fig,(ax0,ax1) = plt.subplots(1,2, figsize=(10,4))
            for h in hooks:
                ms,ss = h.stats
                ax0.plot(ms[:10])
                ax1.plot(ss[:10])
            plt.legend(range(6));
            
            fig,(ax0,ax1) = plt.subplots(1,2, figsize=(10,4))
            for h in hooks:
                ms,ss = h.stats
                ax0.plot(ms)
                ax1.plot(ss)
            plt.legend(range(6));
        ```
    
    * We can add addition statistics such as histogram
        ```python
        def append_stats(hook, mod, inp, outp):
            if not hasattr(hook,'stats'): hook.stats = ([],[],[])
            means,stds,hists = hook.stats
            means.append(outp.data.mean().cpu())
            stds .append(outp.data.std().cpu())
            hists.append(outp.data.cpu().histc(40,0,10)) #histc isn't implemented on the GPU

        with Hooks(model, append_stats) as hooks: run.fit(1, learn) 
        def get_hist(h): return torch.stack(h.stats[2]).t().float().log1p()

        # Plot histogram
        fig,axes = plt.subplots(2,2, figsize=(15,6))
        for ax,h in zip(axes.flatten(), hooks[:4]):
            ax.imshow(get_hist(h), origin='lower')
            ax.axis('off')
        plt.tight_layout()
        ```
    
    * Create generalized ReLU
      * Based on learning from statistics and histogram, we learned there are many activation get really close to 0 (especially in later layers) with default ReLU
      * We can improve default ReLU with leaky, subtract, and max-clipped version
      * We also enable initialization of both kaimin normal and kaimin uniform

        ```python
        def get_cnn_layers(data, nfs, layer, **kwargs):
            nfs = [1] + nfs
            return [layer(nfs[i], nfs[i+1], 5 if i==0 else 3, **kwargs)
                    for i in range(len(nfs)-1)] + [
                nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(nfs[-1], data.c)]

        def conv_layer(ni, nf, ks=3, stride=2, **kwargs):
            return nn.Sequential(
                nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride), GeneralRelu(**kwargs))

        class GeneralRelu(nn.Module):
            def __init__(self, leak=None, sub=None, maxv=None):
                super().__init__()
                self.leak,self.sub,self.maxv = leak,sub,maxv

            def forward(self, x): 
                x = F.leaky_relu(x,self.leak) if self.leak is not None else F.relu(x)
                if self.sub is not None: x.sub_(self.sub)
                if self.maxv is not None: x.clamp_max_(self.maxv)
                return x

        def init_cnn(m, uniform=False):
            f = init.kaiming_uniform_ if uniform else init.kaiming_normal_
            for l in m:
                if isinstance(l, nn.Sequential):
                    f(l[0].weight, a=0.1)
                    l[0].bias.data.zero_()

        def get_cnn_model(data, nfs, layer, **kwargs):
            return nn.Sequential(*get_cnn_layers(data, nfs, layer, **kwargs))
        
        def append_stats(hook, mod, inp, outp):
            if not hasattr(hook,'stats'): hook.stats = ([],[],[])
            means,stds,hists = hook.stats
            means.append(outp.data.mean().cpu())
            stds .append(outp.data.std().cpu())
            hists.append(outp.data.cpu().histc(40,-7,7))
        
        model =  get_cnn_model(data, nfs, conv_layer, leak=0.1, sub=0.4, maxv=6.)
        init_cnn(model)
        learn,run = get_runner(model, data, lr=0.9, cbs=cbfs)

        with Hooks(model, append_stats) as hooks:
            run.fit(1, learn)
            fig,(ax0,ax1) = plt.subplots(1,2, figsize=(10,4))
            for h in hooks:
                ms,ss,hi = h.stats
                ax0.plot(ms[:10])
                ax1.plot(ss[:10])
                h.remove()
            plt.legend(range(5));
            
            fig,(ax0,ax1) = plt.subplots(1,2, figsize=(10,4))
            for h in hooks:
                ms,ss,hi = h.stats
                ax0.plot(ms)
                ax1.plot(ss)
            plt.legend(range(5));
        
        fig,axes = plt.subplots(2,2, figsize=(15,6))
        for ax,h in zip(axes.flatten(), hooks[:4]):
            ax.imshow(get_hist(h), origin='lower')
            ax.axis('off')
        plt.tight_layout()
        ```

* Create a get learner, runner function

    ```python
    def get_learn_run(nfs, data, lr, layer, cbs=None, opt_func=None, uniform=False, **kwargs):
        model = get_cnn_model(data, nfs, layer, **kwargs)
        init_cnn(model, uniform=uniform)
        return get_runner(model, data, lr=lr, cbs=cbs, opt_func=opt_func)

    # Create learning rate schedule
    sched = combine_scheds([0.5, 0.5], [sched_cos(0.2, 1.), sched_cos(1., 0.1)]) 

    learn,run = get_learn_run(nfs, data, 1., conv_layer, cbs=cbfs+[partial(ParamScheduler,'lr', sched)])
    run.fit(8, learn)
    ```

* Batch Normalization and its variation

  * Batch Normlization Math
  
    $\mu = \frac{1}{m}\sum_{i=1}^{m}{x_i}$ mini batch mean

    $\sigma^2 = \frac{1}{m}\sum_{i=1}^{m}{(x_i - \mu)^2}$ mini batch variance

    $\hat{x} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$ Normalization

    $y_i = \gamma\hat{x_i} + \beta$ Scale and shift

  * Batch Normal code

    ```python
    class BatchNorm(nn.Module):
        def __init__(self, nf, mom=0.1, eps=1e-5):
            super().__init__()
            # NB: pytorch bn mom is opposite of what you'd expect
            self.mom,self.eps = mom,eps
            self.mults = nn.Parameter(torch.ones (nf,1,1))
            self.adds  = nn.Parameter(torch.zeros(nf,1,1))
            self.register_buffer('vars',  torch.ones(1,nf,1,1))
            self.register_buffer('means', torch.zeros(1,nf,1,1))

        def update_stats(self, x):
            m = x.mean((0,2,3), keepdim=True)
            v = x.var ((0,2,3), keepdim=True)
            self.means.lerp_(m, self.mom)
            self.vars.lerp_ (v, self.mom)
            return m,v
            
        def forward(self, x):
            if self.training:
                with torch.no_grad(): m,v = self.update_stats(x)
            else: m,v = self.means,self.vars
            x = (x-m) / (v+self.eps).sqrt()
            return x*self.mults + self.adds
    ```
    * Pytorch batch norm `nn.BatchNorm2d()`
    ```python
    def conv_layer(ni, nf, ks=3, stride=2, bn=True, **kwargs):
        # No bias needed if using bn
        layers = [nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride, bias=not bn),
                GeneralRelu(**kwargs)]
        # if bn: layers.append(BatchNorm(nf)) => custom
        if bn: layers.append(nn.BatchNorm2d(nf, eps=1e-5, momentum=0.1))
        return nn.Sequential(*layers)
    
    def init_cnn_(m, f):
        if isinstance(m, nn.Conv2d):
            f(m.weight, a=0.1)
            if getattr(m, 'bias', None) is not None: m.bias.data.zero_()
        for l in m.children(): init_cnn_(l, f)

    def init_cnn(m, uniform=False):
        f = init.kaiming_uniform_ if uniform else init.kaiming_normal_
        init_cnn_(m, f)

    def get_learn_run(nfs, data, lr, layer, cbs=None, opt_func=None, uniform=False, **kwargs):
        model = get_cnn_model(data, nfs, layer, **kwargs)
        init_cnn(model, uniform=uniform)
        return get_runner(model, data, lr=lr, cbs=cbs, opt_func=opt_func)
    
    learn,run = get_learn_run(nfs, data, 0.9, conv_layer, cbs=cbfs)
    ```
    
    * Other normalization is needed because batch normalization cannot be applied to online learning tasks or to extremely large distributed models where the minibatches have to be small. The problem: When we compute the statistics (mean and std) for a BatchNorm Layer on a small batch, it is possible that we get a standard deviation very close to 0. because there aren't many samples (the variance of one thing is 0. since it's equal to its mean)
      * Layer norm
      * Instance norm
      * Group norm
      * FastAI's unique: **Running Batch Norm** (uses smoother running mean and variance for the mean and std.)


# Lecture 11 Data Block API, custom optimizer

* Data block API

    We need an efficient way ot load and train images data if the datasets is larger than the computer RAM capacity
    Steps to do
    * Get files
    * Split validation set
    * random%, folder name, csv, ...
    * Label:
    * folder name, file name/re, csv, ...
    * Transform per image (optional)
    * Transform to tensor
    * DataLoader
    * Transform per batch (optional)
    * DataBunch
    * Add test set (optional)

  * Get an item list

    ```python
    import PIL,os,mimetypes

    # If Path is an pathlib object
    # e.g path = PosixPath('/home/ubuntu/.fastai/data/imagenette-160')
    # We can create simply lambda function for pathlib object to list files
    Path.ls = lambda x: list(x.iterdir())
    
    # to call
    path.ls()
    ```
  * Get files function

    `os.walk` and `os.scandir` is fastest way to traverse files in directory. `os.walk` walk the diretories recursively

    ```python
    # Get all image file extension
    image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))

    # A function to create sets (faster than list)
    def setify(o): return o if isinstance(o,set) else set(listify(o))

    def _get_files(p, fs, extensions=None):
        p = Path(p)
        res = [p/f for f in fs if not f.startswith('.')
            and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
        return res
    

    def get_files(path, extensions=None, recurse=False, include=None):
        path = Path(path)
        extensions = setify(extensions)
        extensions = {e.lower() for e in extensions}
        if recurse:
            res = []
            for i,(p,d,f) in enumerate(os.walk(path)): # returns (dirpath, dirnames, filenames)
                if include is not None and i==0: d[:] = [o for o in d if o in include]
                else:                            d[:] = [o for o in d if not o.startswith('.')]
                res += _get_files(p, f, extensions)
            return res
        else:
            f = [o.name for o in os.scandir(path) if o.is_file()]
            return _get_files(path, f, extensions)
    ```

    
    Use the `ListContainer` class to store our objects in an `ItemList`. The get method will need to be subclassed to explain how to access an element (open an image for instance), then the private `_get` method can allow us to apply any additional transform to it.

    `new` will be used in conjunction with `__getitem__` (that works for one index or a list of indices) to create training and validation set from a single stream when we split the data.

    ```python
    # Compose function chains a list of functions together with an optional _order fastai attribute 
    def compose(x, funcs, *args, order_key='_order', **kwargs):
        key = lambda o: getattr(o, order_key, 0)
        for f in sorted(listify(funcs), key=key): x = f(x, **kwargs)
        return x

    # inherit ListContainer class but add _get() function to add transformer function using compose()
    class ItemList(ListContainer):
        def __init__(self, items, path='.', tfms=None):
            super().__init__(items)
            self.path,self.tfms = Path(path),tfms

        def __repr__(self): return f'{super().__repr__()}\nPath: {self.path}'
        
        def new(self, items, cls=None):
            if cls is None: cls=self.__class__
            return cls(items, self.path, tfms=self.tfms)
        
        def  get(self, i): return i
        def _get(self, i): return compose(self.get(i), self.tfms)
        
        def __getitem__(self, idx):
            res = super().__getitem__(idx)
            if isinstance(res,list): return [self._get(o) for o in res]
            return self._get(res)

    # inherit Itemlist function to allow PIL to open image and open from a path using get_fines()
    class ImageList(ItemList):
        @classmethod
        def from_files(cls, path, extensions=None, recurse=True, include=None, **kwargs):
            if extensions is None: extensions = image_extensions
            return cls(get_files(path, extensions, recurse=recurse, include=include), path, **kwargs)
        
        def get(self, fn): return PIL.Image.open(fn)
    ```

    From above we can define transformers to process image

    ```python
    class Transform(): _order=0

    class MakeRGB(Transform):
        def __call__(self, item): return item.convert('RGB')

    def make_rgb(item): return item.convert('RGB')

    # example call
    image_list1 = ImageList.from_files(path, tfms=make_rgb)
    make_rgb = MakeRGB()
    image_list2 = ImageList.from_files(path, tfms=make_rgb)

    # with __getitem__ we can index into the list
    img = image_list2[0] 
    ```

  * splitter to split based on diretory
    ```python
    def grandparent_splitter(fn, valid_name='valid', train_name='train'):
        gp = fn.parent.parent.name
        return True if gp==valid_name else False if gp==train_name else None

    def split_by_func(items, f):
        mask = [f(o) for o in items]
        # `None` values will be filtered out
        f = [o for o,m in zip(items,mask) if m==False]
        t = [o for o,m in zip(items,mask) if m==True ]
        return f,t
    
    splitter = partial(grandparent_splitter, valid_name='val')
    train,valid = split_by_func(il, splitter)
    ```

    Create a class for splitter
    ```python
    class SplitData():
        def __init__(self, train, valid): self.train,self.valid = train,valid
        
        def __getattr__(self,k): return getattr(self.train,k)
        #This is needed if we want to pickle SplitData and be able to load it back without recursion errors
        def __setstate__(self,data:Any): self.__dict__.update(data) 
        
        @classmethod
        def split_by_func(cls, il, f):
            lists = map(il.new, split_by_func(il.items, f))
            return cls(*lists)

        def __repr__(self): return f'{self.__class__.__name__}\nTrain: {self.train}\nValid: {self.valid}\n'
    ```

  * Labelling

    Labeling has to be done *after* splitting, because it uses *training* set information to apply to the *validation* set, using a *Processor*.

    A *Processor* is a transformation that is applied to all the inputs once at initialization, with some *state* computed on the training set that is then applied without modification on the validation set (and maybe the test set or at inference time on a single item). For instance, it could be **processing texts** to **tokenize**, then **numericalize** them. In that case we want the validation set to be numericalized with exactly the same vocabulary as the training set.

    Another example is in **tabular data**, where we **fill missing values** with (for instance) the median computed on the training set. That statistic is stored in the inner state of the *Processor* and applied on the validation set.

    In our case, we want to **convert label strings to numbers** in a consistent and reproducible way. So we create a list of possible labels in the training set, and then convert our labels to numbers based on this *vocab*.

    ```python
    from collections import OrderedDict

    def uniqueify(x, sort=False):
        res = list(OrderedDict.fromkeys(x).keys())
        if sort: res.sort()
        return res
    
    class Processor(): 
        def process(self, items): return items

    class CategoryProcessor(Processor):
        def __init__(self): self.vocab=None
        
        def __call__(self, items):
            #The vocab is defined on the first use.
            if self.vocab is None:
                self.vocab = uniqueify(items)
                self.otoi  = {v:k for k,v in enumerate(self.vocab)}
            return [self.proc1(o) for o in items]
        def proc1(self, item):  return self.otoi[item]
        
        def deprocess(self, idxs):
            assert self.vocab is not None
            return [self.deproc1(idx) for idx in idxs]
        def deproc1(self, idx): return self.vocab[idx]
    
    def parent_labeler(fn): return fn.parent.name

    def _label_by_func(ds, f, cls=ItemList): return cls([f(o) for o in ds.items], path=ds.path)

    class LabeledData():
        def process(self, il, proc): return il.new(compose(il.items, proc))

        def __init__(self, x, y, proc_x=None, proc_y=None):
            self.x,self.y = self.process(x, proc_x),self.process(y, proc_y)
            self.proc_x,self.proc_y = proc_x,proc_y
            
        def __repr__(self): return f'{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n'
        def __getitem__(self,idx): return self.x[idx],self.y[idx]
        def __len__(self): return len(self.x)
        
        def x_obj(self, idx): return self.obj(self.x, idx, self.proc_x)
        def y_obj(self, idx): return self.obj(self.y, idx, self.proc_y)
        
        def obj(self, items, idx, procs):
            isint = isinstance(idx, int) or (isinstance(idx,torch.LongTensor) and not idx.ndim)
            item = items[idx]
            for proc in reversed(listify(procs)):
                item = proc.deproc1(item) if isint else proc.deprocess(item)
            return item

        @classmethod
        def label_by_func(cls, il, f, proc_x=None, proc_y=None):
            return cls(il, _label_by_func(il, f), proc_x=proc_x, proc_y=proc_y)

    def label_by_func(sd, f, proc_x=None, proc_y=None):
        train = LabeledData.label_by_func(sd.train, f, proc_x=proc_x, proc_y=proc_y)
        valid = LabeledData.label_by_func(sd.valid, f, proc_x=proc_x, proc_y=proc_y)
        return SplitData(train,valid)
    ```

    The first time, we create vocab for training, validation will use training set's vocab to ensure the label works correctly

* Transform Tensors
we convert the image to a by tensor before converting it to float and dividing by 255
    ```python
    class ResizeFixed(Transform):
        _order=10
        def __init__(self,size):
            if isinstance(size,int): size=(size,size)
            self.size = size
            
        def __call__(self, item): return item.resize(self.size, PIL.Image.BILINEAR)

    def to_byte_tensor(item):
        res = torch.ByteTensor(torch.ByteStorage.from_buffer(item.tobytes()))
        w,h = item.size
        return res.view(h,w,-1).permute(2,0,1)

    to_byte_tensor._order=20

    def to_float_tensor(item): return item.float().div_(255.)

    to_float_tensor._order=30

    tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]

    il = ImageList.from_files(path, tfms=tfms)
    sd = SplitData.split_by_func(il, splitter)
    ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())

    def show_image(im, figsize=(3,3)):
        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.imshow(im.permute(1,2,0))
    ```

* Define Data Bunch and put thing together
    ```python
    class DataBunch():
        def __init__(self, train_dl, valid_dl, c_in=None, c_out=None):
            self.train_dl,self.valid_dl,self.c_in,self.c_out = train_dl,valid_dl,c_in,c_out

        @property
        def train_ds(self): return self.train_dl.dataset

        @property
        def valid_ds(self): return self.valid_dl.dataset

    def databunchify(sd, bs, c_in=None, c_out=None, **kwargs):
        dls = get_dls(sd.train, sd.valid, bs, **kwargs)
        return DataBunch(*dls, c_in=c_in, c_out=c_out)


    SplitData.to_databunch = databunchify

    path = datasets.untar_data(datasets.URLs.IMAGENETTE_160)
    tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]

    il = ImageList.from_files(path, tfms=tfms)
    sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))
    ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())
    data = ll.to_databunch(bs, c_in=3, c_out=10, num_workers=4)
    ```

* Define model and call batch
    ```python
    cbfs = [partial(AvgStatsCallback,accuracy), CudaCallback]

    # normalize multiple channels
    def normalize_chan(x, mean, std):
        return (x-mean[...,None,None]) / std[...,None,None]
    cbfs.append(partial(BatchTransformXCallback, norm_imagenette))

    import math
    def prev_pow_2(x): return 2**math.floor(math.log2(x))

    def get_cnn_layers(data, nfs, layer, **kwargs):
        def f(ni, nf, stride=2): return layer(ni, nf, 3, stride=stride, **kwargs)
        l1 = data.c_in
        l2 = prev_pow_2(l1*3*3)
        layers =  [f(l1  , l2  , stride=1),
                f(l2  , l2*2, stride=2),
                f(l2*2, l2*4, stride=2)]
        nfs = [l2*4] + nfs
        layers += [f(nfs[i], nfs[i+1]) for i in range(len(nfs)-1)]
        layers += [nn.AdaptiveAvgPool2d(1), Lambda(flatten), 
                nn.Linear(nfs[-1], data.c_out)]
        return layers

    def get_cnn_model(data, nfs, layer, **kwargs):
        return nn.Sequential(*get_cnn_layers(data, nfs, layer, **kwargs))

    def get_learn_run(nfs, data, lr, layer, cbs=None, opt_func=None, **kwargs):
        model = get_cnn_model(data, nfs, layer, **kwargs)
        init_cnn(model)
        return get_runner(model, data, lr=lr, cbs=cbs, opt_func=opt_func)

    sched = combine_scheds([0.3,0.7], cos_1cycle_anneal(0.1,0.3,0.05))
    learn,run = get_learn_run(nfs, data, 0.2, conv_layer, cbs=cbfs+[
        partial(ParamScheduler, 'lr', sched)
    ])

    def model_summary(run, learn, data, find_all=False):
        xb,yb = get_batch(data.valid_dl, run)
        device = next(learn.model.parameters()).device#Model may not be on the GPU yet
        xb,yb = xb.to(device),yb.to(device)
        mods = find_modules(learn.model, is_lin_layer) if find_all else learn.model.children()
        f = lambda hook,mod,inp,out: print(f"{mod}\n{out.shape}\n")
        with Hooks(mods, f) as hooks: learn.model(xb)
    model_summary(run, learn, data)
    ```

* Customize optimizer

    We can create one optmizer that can be program to become many optimizers
    * We need to access to all the parameters (weight tensor and bias tensors)
      * `model.params` can access model parameters
      * We can use a list of list (`param_group`) so we can access params of different layers
      * Each `param_group` can have its own hyper-parameters as a dictionary
    * We create a stepper for different optmizer types
    * Based on the above, we can create a class that takes `model.params`, `stepper` as class input and create an optimizer based on the properties of `params` and `stepper`

    ```python
    def maybe_update(os, dest, f):
        for o in os:
            for k,v in f(o).items():
                if k not in dest: dest[k] = v

    def get_defaults(d): return getattr(d,'_defaults',{})

    class Optimizer():
        def __init__(self, params, steppers, **defaults):
            self.steppers = listify(steppers)
            maybe_update(self.steppers, defaults, get_defaults)
            # might be a generator
            self.param_groups = list(params)
            # ensure params is a list of lists
            if not isinstance(self.param_groups[0], list): self.param_groups = [self.param_groups]
            self.hypers = [{**defaults} for p in self.param_groups]

        def grad_params(self):
            return [(p,hyper) for pg,hyper in zip(self.param_groups,self.hypers)
                for p in pg if p.grad is not None]

        def zero_grad(self):
            for p,hyper in self.grad_params():
                p.grad.detach_()
                p.grad.zero_()

        def step(self):
            for p,hyper in self.grad_params(): compose(p, self.steppers, **hyper)
    
    # Create an SGD optimizer
    def sgd_step(p, lr, **kwargs):
        p.data.add_(-lr, p.grad.data)
        return p
    
    opt_func = partial(Optimizer, steppers=[sgd_step])
    ```

    Redefine the callbacks that were using properties from the PyTorch optimizer
    ```python
    class Recorder(Callback):
        def begin_fit(self): self.lrs,self.losses = [],[]

        def after_batch(self):
            if not self.in_train: return
            self.lrs.append(self.opt.hypers[-1]['lr'])
            self.losses.append(self.loss.detach().cpu())        

        def plot_lr  (self): plt.plot(self.lrs)
        def plot_loss(self): plt.plot(self.losses)
            
        def plot(self, skip_last=0):
            losses = [o.item() for o in self.losses]
            n = len(losses)-skip_last
            plt.xscale('log')
            plt.plot(self.lrs[:n], losses[:n])

    class ParamScheduler(Callback):
        _order=1
        def __init__(self, pname, sched_funcs):
            self.pname,self.sched_funcs = pname,listify(sched_funcs)

        def begin_batch(self): 
            if not self.in_train: return
            fs = self.sched_funcs
            if len(fs)==1: fs = fs*len(self.opt.param_groups)
            pos = self.n_epochs/self.epochs
            for f,h in zip(fs,self.opt.hypers): h[self.pname] = f(pos)
                
    class LR_Find(Callback):
        _order=1
        def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
            self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr
            self.best_loss = 1e9
            
        def begin_batch(self): 
            if not self.in_train: return
            pos = self.n_iter/self.max_iter
            lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
            for pg in self.opt.hypers: pg['lr'] = lr
                
        def after_step(self):
            if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:
                raise CancelTrainException()
            if self.loss < self.best_loss: self.best_loss = self.loss
    ```

    We can now use other stepper such as `weight decay` or `l2 regularization`
    ```python
    #export
    def weight_decay(p, lr, wd, **kwargs):
        p.data.mul_(1 - lr*wd)
        return p
    weight_decay._defaults = dict(wd=0.)

    def l2_reg(p, lr, wd, **kwargs):
        p.grad.data.add_(wd, p.data)
        return p
    l2_reg._defaults = dict(wd=0.)
    ```

    Another variation of optimizer is stateful optimizer or Momentum
    * Using a class variable of dict to store state
    ```python
    class StatefulOptimizer(Optimizer):
        def __init__(self, params, steppers, stats=None, **defaults): 
            self.stats = listify(stats)
            maybe_update(self.stats, defaults, get_defaults)
            super().__init__(params, steppers, **defaults)
            self.state = {}
            
        def step(self):
            for p,hyper in self.grad_params():
                if p not in self.state:
                    #Create a state for p and call all the statistics to initialize it.
                    self.state[p] = {}
                    maybe_update(self.stats, self.state[p], lambda o: o.init_state(p))
                state = self.state[p]
                for stat in self.stats: state = stat.update(p, state, **hyper)
                compose(p, self.steppers, **state, **hyper)
                self.state[p] = state

    class Stat():
        _defaults = {}
        def init_state(self, p): raise NotImplementedError
        def update(self, p, state, **kwargs): raise NotImplementedError        
    
    class AverageGrad(Stat):
        _defaults = dict(mom=0.9)

        def init_state(self, p): return {'grad_avg': torch.zeros_like(p.grad.data)}
        def update(self, p, state, mom, **kwargs):
            state['grad_avg'].mul_(mom).add_(p.grad.data)
            return state

    def momentum_step(p, lr, grad_avg, **kwargs):
        p.data.add_(-lr, grad_avg)
        return p
    
    sgd_mom_opt = partial(StatefulOptimizer, steppers=[momentum_step,weight_decay],
                  stats=AverageGrad(), wd=0.01)
    learn,run = get_learn_run(nfs, data, 0.3, conv_layer, cbs=cbfs, opt_func=sgd_mom_opt)
    run.fit(1, learn)
    ```

    Adam: dampen debiased momentum divided by dampen debiased root sum of square gradient
    ```python
    class AverageGrad(Stat):
        _defaults = dict(mom=0.9)
        
        def __init__(self, dampening:bool=False): self.dampening=dampening
        def init_state(self, p): return {'grad_avg': torch.zeros_like(p.grad.data)}
        def update(self, p, state, mom, **kwargs):
            state['mom_damp'] = 1-mom if self.dampening else 1.
            state['grad_avg'].mul_(mom).add_(state['mom_damp'], p.grad.data)
            return state

    class AverageSqrGrad(Stat):
        _defaults = dict(sqr_mom=0.99)
        
        def __init__(self, dampening:bool=True): self.dampening=dampening
        def init_state(self, p): return {'sqr_avg': torch.zeros_like(p.grad.data)}
        def update(self, p, state, sqr_mom, **kwargs):
            state['sqr_damp'] = 1-sqr_mom if self.dampening else 1.
            state['sqr_avg'].mul_(sqr_mom).addcmul_(state['sqr_damp'], p.grad.data, p.grad.data)
            return state
    
    class StepCount(Stat):
        def init_state(self, p): return {'step': 0}
        def update(self, p, state, **kwargs):
            state['step'] += 1
            return state
    
    def debias(mom, damp, step): return damp * (1 - mom**step) / (1-mom)
    def adam_step(p, lr, mom, mom_damp, step, sqr_mom, sqr_damp, grad_avg, sqr_avg, eps, **kwargs):
        debias1 = debias(mom,     mom_damp, step)
        debias2 = debias(sqr_mom, sqr_damp, step)
        p.data.addcdiv_(-lr / debias1, grad_avg, (sqr_avg/debias2).sqrt() + eps)
        return p
    adam_step._defaults = dict(eps=1e-5)

    def adam_opt(xtra_step=None, **kwargs):
        return partial(StatefulOptimizer, steppers=[adam_step,weight_decay]+listify(xtra_step),
                    stats=[AverageGrad(dampening=True), AverageSqrGrad(), StepCount()], **kwargs)
    
    learn,run = get_learn_run(nfs, data, 0.001, conv_layer, cbs=cbfs, opt_func=adam_opt())
    run.fit(3, learn)
    ```

    LAMB Optimizer
    debiased, dampened, exponential weighted moving average over a layer

    ```python
    def lamb_step(p, lr, mom, mom_damp, step, sqr_mom, sqr_damp, grad_avg, sqr_avg, eps, wd, **kwargs):
        debias1 = debias(mom,     mom_damp, step)
        debias2 = debias(sqr_mom, sqr_damp, step)
        r1 = p.data.pow(2).mean().sqrt()
        step = (grad_avg/debias1) / ((sqr_avg/debias2).sqrt()+eps) + wd*p.data
        r2 = step.pow(2).mean().sqrt()
        p.data.add_(-lr * min(r1/r2,10), step)
        return p
    lamb_step._defaults = dict(eps=1e-6, wd=0.)
    lamb = partial(StatefulOptimizer, steppers=lamb_step, stats=[AverageGrad(dampening=True), AverageSqrGrad(), StepCount()])
    learn,run = get_learn_run(nfs, data, 0.003, conv_layer, cbs=cbfs, opt_func=lamb)
    run.fit(3, learn)
    ```

* Optimizer Learner and add progress bar
```python
def param_getter(m): return m.parameters()

class Learner():
    def __init__(self, model, data, loss_func, opt_func=sgd_opt, lr=1e-2, splitter=param_getter,
                 cbs=None, cb_funcs=None):
        self.model,self.data,self.loss_func,self.opt_func,self.lr,self.splitter = model,data,loss_func,opt_func,lr,splitter
        self.in_train,self.logger,self.opt = False,print,None
        
        # NB: Things marked "NEW" are covered in lesson 12
        # NEW: avoid need for set_runner
        self.cbs = []
        self.add_cb(TrainEvalCallback())
        self.add_cbs(cbs)
        self.add_cbs(cbf() for cbf in listify(cb_funcs))

    def add_cbs(self, cbs):
        for cb in listify(cbs): self.add_cb(cb)
            
    def add_cb(self, cb):
        cb.set_runner(self)
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def remove_cbs(self, cbs):
        for cb in listify(cbs): self.cbs.remove(cb)
            
    def one_batch(self, i, xb, yb):
        try:
            self.iter = i
            self.xb,self.yb = xb,yb;                        self('begin_batch')
            self.pred = self.model(self.xb);                self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb); self('after_loss')
            if not self.in_train: return
            self.loss.backward();                           self('after_backward')
            self.opt.step();                                self('after_step')
            self.opt.zero_grad()
        except CancelBatchException:                        self('after_cancel_batch')
        finally:                                            self('after_batch')

    def all_batches(self):
        self.iters = len(self.dl)
        try:
            for i,(xb,yb) in enumerate(self.dl): self.one_batch(i, xb, yb)
        except CancelEpochException: self('after_cancel_epoch')

    def do_begin_fit(self, epochs):
        self.epochs,self.loss = epochs,tensor(0.)
        self('begin_fit')

    def do_begin_epoch(self, epoch):
        self.epoch,self.dl = epoch,self.data.train_dl
        return self('begin_epoch')

    def fit(self, epochs, cbs=None, reset_opt=False):
        # NEW: pass callbacks to fit() and have them removed when done
        self.add_cbs(cbs)
        # NEW: create optimizer on fit(), optionally replacing existing
        if reset_opt or not self.opt: self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)
            
        try:
            self.do_begin_fit(epochs)
            for epoch in range(epochs):
                if not self.do_begin_epoch(epoch): self.all_batches()

                with torch.no_grad(): 
                    self.dl = self.data.valid_dl
                    if not self('begin_validate'): self.all_batches()
                self('after_epoch')
            
        except CancelTrainException: self('after_cancel_train')
        finally:
            self('after_fit')
            self.remove_cbs(cbs)

    ALL_CBS = {'begin_batch', 'after_pred', 'after_loss', 'after_backward', 'after_step',
        'after_cancel_batch', 'after_batch', 'after_cancel_epoch', 'begin_fit',
        'begin_epoch', 'begin_validate', 'after_epoch',
        'after_cancel_train', 'after_fit'}
    
    def __call__(self, cb_name):
        res = False
        assert cb_name in self.ALL_CBS
        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) and res
        return res

class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)
    
    def begin_fit(self):
        met_names = ['loss'] + [m.__name__ for m in self.train_stats.metrics]
        names = ['epoch'] + [f'train_{n}' for n in met_names] + [
            f'valid_{n}' for n in met_names] + ['time']
        self.logger(names)
    
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()
        
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)
    
    def after_epoch(self):
        stats = [str(self.epoch)] 
        for o in [self.train_stats, self.valid_stats]:
            stats += [f'{v:.6f}' for v in o.avg_stats] 
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)

class ProgressCallback(Callback):
    _order=-1
    def begin_fit(self):
        self.mbar = master_bar(range(self.epochs))
        self.mbar.on_iter_begin()
        self.run.logger = partial(self.mbar.write, table=True)
        
    def after_fit(self): self.mbar.on_iter_end()
    def after_batch(self): self.pb.update(self.iter)
    def begin_epoch   (self): self.set_pb()
    def begin_validate(self): self.set_pb()
        
    def set_pb(self):
        self.pb = progress_bar(self.dl, parent=self.mbar)
        self.mbar.update(self.epoch)

cbfs = [partial(AvgStatsCallback,accuracy),
        CudaCallback,
        ProgressCallback,
        partial(BatchTransformXCallback, norm_imagenette)]

def get_learner(nfs, data, lr, layer, loss_func=F.cross_entropy,
                cb_funcs=None, opt_func=sgd_opt, **kwargs):
    model = get_cnn_model(data, nfs, layer, **kwargs)
    init_cnn(model)
    return Learner(model, data, loss_func, lr=lr, cb_funcs=cb_funcs, opt_func=opt_func)

learn = get_learner(nfs, data, 0.4, conv_layer, cb_funcs=cbfs)
```

* Data Augmentation
  * Important to visualize your augmentation to make sense of it: does it impact or help what we want to learn?
  * Techniques:
    * Flip
    * Random Crop
    * Random resize crop
    * prespective warping

# Lecture 12 Advanced Training Techniques; ULMFIT


* Mixup is mix up (linear combination) of two images as an augmentation approach, but we also need to do linear combination of labels, linear combination of loss function

    ```python
    class NoneReduce():
        def __init__(self, loss_func): 
            self.loss_func,self.old_red = loss_func,None
            
        def __enter__(self):
            if hasattr(self.loss_func, 'reduction'):
                self.old_red = getattr(self.loss_func, 'reduction')
                setattr(self.loss_func, 'reduction', 'none')
                return self.loss_func
            else: return partial(self.loss_func, reduction='none')
            
        def __exit__(self, type, value, traceback):
            if self.old_red is not None: setattr(self.loss_func, 'reduction', self.old_red)    

    from torch.distributions.beta import Beta

    def unsqueeze(input, dims):
        for dim in listify(dims): input = torch.unsqueeze(input, dim)
        return input

    def reduce_loss(loss, reduction='mean'):
        return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss    

    class MixUp(Callback):
        _order = 90 #Runs after normalization and cuda
        def __init__(self, α:float=0.4): self.distrib = Beta(tensor([α]), tensor([α]))
        
        def begin_fit(self): self.old_loss_func,self.run.loss_func = self.run.loss_func,self.loss_func
        
        def begin_batch(self):
            if not self.in_train: return #Only mixup things during training
            λ = self.distrib.sample((self.yb.size(0),)).squeeze().to(self.xb.device)
            λ = torch.stack([λ, 1-λ], 1)
            self.λ = unsqueeze(λ.max(1)[0], (1,2,3))
            shuffle = torch.randperm(self.yb.size(0)).to(self.xb.device)
            xb1,self.yb1 = self.xb[shuffle],self.yb[shuffle]
            self.run.xb = lin_comb(self.xb, xb1, self.λ)
            
        def after_fit(self): self.run.loss_func = self.old_loss_func
        
        def loss_func(self, pred, yb):
            if not self.in_train: return self.old_loss_func(pred, yb)
            with NoneReduce(self.old_loss_func) as loss_func:
                loss1 = loss_func(pred, yb)
                loss2 = loss_func(pred, self.yb1)
            loss = lin_comb(loss1, loss2, self.λ)
            return reduce_loss(loss, getattr(self.old_loss_func, 'reduction', 'mean'))

    cbfs = [partial(AvgStatsCallback,accuracy),
        CudaCallback, 
        ProgressCallback,
        partial(BatchTransformXCallback, norm_imagenette),
        MixUp]
    
    ```

* Label smoothing
Another regularization technique that's often used is label smoothing. It's designed to make the model a little bit less certain of it's decision by changing a little bit its target: instead of wanting to predict 1 for the correct class and 0 for all the others, we ask it to predict `1-ε` for the correct class and `ε` for all the others, with `ε` a (small) positive number and N the number of classes. This can be written as:

$$loss = (1-ε) ce(i) + ε \sum ce(j) / N$$

where `ce(x)` is cross-entropy of `x` (i.e. $-\log(p_{x})$), and `i` is the correct class. This can be coded in a loss function:

    ```python
    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, ε:float=0.1, reduction='mean'):
            super().__init__()
            self.ε,self.reduction = ε,reduction
        
        def forward(self, output, target):
            c = output.size()[-1]
            log_preds = F.log_softmax(output, dim=-1)
            loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
            nll = F.nll_loss(log_preds, target, reduction=self.reduction)
            return lin_comb(loss/c, nll, self.ε)
    
    learn = get_learner(nfs, data, 0.4, conv_layer, cb_funcs=cbfs, loss_func=LabelSmoothingCrossEntropy())
    ```

* ULMFit is transfer learning applied to AWD-LSTM
  * LSTM vs Transformer
    * LSTM has states, transformer do not
    * LSTM is more efficient, transformer has to learn the neighbor all the time
  * Language model is a general term that describe we are predicting the next item in the sequence
  * Jeremy think stopwords, stemming, lemmatization is not needed in deep learning. They contains useful information about language

* NLP Pipeline
  * Read Data from files
  ```python
  def read_file(fn): 
    with open(fn, 'r', encoding = 'utf8') as f: return f.read()
    
    class TextList(ItemList):
        @classmethod
        def from_files(cls, path, extensions='.txt', recurse=True, include=None, **kwargs):
            return cls(get_files(path, extensions, recurse=recurse, include=include), path, **kwargs)
        
        def get(self, i):
            if isinstance(i, Path): return read_file(i)
            return i
  ```

  * Tokenization

    Preprocessing

    ```python
    import spacy, html

    UNK, PAD, BOS, EOS, TK_REP, TK_WREP, TK_UP, TK_MAJ = "xxunk xxpad xxbos xxeos xxrep xxwrep xxup xxmaj".split()

    def sub_br(t):
        "Replaces the <br /> by \n"
        re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
        return re_br.sub("\n", t)

    def spec_add_spaces(t):
        "Add spaces around / and #"
        return re.sub(r'([/#])', r' \1 ', t)

    def rm_useless_spaces(t):
        "Remove multiple spaces"
        return re.sub(' {2,}', ' ', t)

    def replace_rep(t):
        "Replace repetitions at the character level: cccc -> TK_REP 4 c"
        def _replace_rep(m:Collection[str]) -> str:
            c,cc = m.groups()
            return f' {TK_REP} {len(cc)+1} {c} '
        re_rep = re.compile(r'(\S)(\1{3,})')
        return re_rep.sub(_replace_rep, t)
        
    def replace_wrep(t):
        "Replace word repetitions: word word word -> TK_WREP 3 word"
        def _replace_wrep(m:Collection[str]) -> str:
            c,cc = m.groups()
            return f' {TK_WREP} {len(cc.split())+1} {c} '
        re_wrep = re.compile(r'(\b\w+\W+)(\1{3,})')
        return re_wrep.sub(_replace_wrep, t)

    def fixup_text(x):
        "Various messy things we've seen in documents"
        re1 = re.compile(r'  +')
        x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
            'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
            '<br />', "\n").replace('\\"', '"').replace('<unk>',UNK).replace(' @.@ ','.').replace(
            ' @-@ ','-').replace('\\', ' \\ ')
        return re1.sub(' ', html.unescape(x))
        
    default_pre_rules = [fixup_text, replace_rep, replace_wrep, spec_add_spaces, rm_useless_spaces, sub_br]
    default_spec_tok = [UNK, PAD, BOS, EOS, TK_REP, TK_WREP, TK_UP, TK_MAJ]

    def replace_all_caps(x):
        "Replace tokens in ALL CAPS by their lower version and add `TK_UP` before."
        res = []
        for t in x:
            if t.isupper() and len(t) > 1: res.append(TK_UP); res.append(t.lower())
            else: res.append(t)
        return res

    def deal_caps(x):
        "Replace all Capitalized tokens in by their lower version and add `TK_MAJ` before."
        res = []
        for t in x:
            if t == '': continue
            if t[0].isupper() and len(t) > 1 and t[1:].islower(): res.append(TK_MAJ)
            res.append(t.lower())
        return res

    def add_eos_bos(x): return [BOS] + x + [EOS]

    default_post_rules = [deal_caps, replace_all_caps, add_eos_bos]
    ```

    Tokenization
    ```python
    from spacy.symbols import ORTH
    from concurrent.futures import ProcessPoolExecutor

    def parallel(func, arr, max_workers=4):
        if max_workers<2: results = list(progress_bar(map(func, enumerate(arr)), total=len(arr)))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                return list(progress_bar(ex.map(func, enumerate(arr)), total=len(arr)))
        if any([o is not None for o in results]): return results

    class TokenizeProcessor(Processor):
    def __init__(self, lang="en", chunksize=2000, pre_rules=None, post_rules=None, max_workers=4): 
        self.chunksize,self.max_workers = chunksize,max_workers
        self.tokenizer = spacy.blank(lang).tokenizer
        for w in default_spec_tok:
            self.tokenizer.add_special_case(w, [{ORTH: w}])
        self.pre_rules  = default_pre_rules  if pre_rules  is None else pre_rules
        self.post_rules = default_post_rules if post_rules is None else post_rules

    def proc_chunk(self, args):
        i,chunk = args
        chunk = [compose(t, self.pre_rules) for t in chunk]
        docs = [[d.text for d in doc] for doc in self.tokenizer.pipe(chunk)]
        docs = [compose(t, self.post_rules) for t in docs]
        return docs

    def __call__(self, items): 
        toks = []
        if isinstance(items[0], Path): items = [read_file(i) for i in items]
        chunks = [items[i: i+self.chunksize] for i in (range(0, len(items), self.chunksize))]
        toks = parallel(self.proc_chunk, chunks, max_workers=self.max_workers)
        return sum(toks, [])
    
    def proc1(self, item): return self.proc_chunk([item])[0]
    
    def deprocess(self, toks): return [self.deproc1(tok) for tok in toks]
    def deproc1(self, tok):    return " ".join(tok)
    ```

  * Numericalizing
```python
import collections

class NumericalizeProcessor(Processor):
    def __init__(self, vocab=None, max_vocab=60000, min_freq=2): 
        self.vocab,self.max_vocab,self.min_freq = vocab,max_vocab,min_freq
    
    def __call__(self, items):
        #The vocab is defined on the first use.
        if self.vocab is None:
            freq = Counter(p for o in items for p in o)
            self.vocab = [o for o,c in freq.most_common(self.max_vocab) if c >= self.min_freq]
            for o in reversed(default_spec_tok):
                if o in self.vocab: self.vocab.remove(o)
                self.vocab.insert(0, o)
        if getattr(self, 'otoi', None) is None:
            self.otoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.vocab)}) 
        return [self.proc1(o) for o in items]
    def proc1(self, item):  return [self.otoi[o] for o in item]
    
    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deproc1(idx) for idx in idxs]
    def deproc1(self, idx): return [self.vocab[i] for i in idx]

```

## [Reference Git](https://github.com/fastai/course-v3.git)
