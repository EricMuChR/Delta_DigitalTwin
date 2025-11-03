import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
import os

class MAMLRegressor:
    def __init__(self, settings, device='cpu'):
        self.configure_device(device)
        self.settings = settings
        self.meta_lr = settings['meta_lr']
        self.inner_lr = settings['inner_lr']
        self.inner_steps = settings['inner_steps']
        self.meta_params = self._init_params()
        self.optimizer = optax.adam(learning_rate=self.meta_lr)
        self.opt_state = self.optimizer.init(self.meta_params)
        print("MAML Regressor 初始化，内循环引擎为Adam。")

    def configure_device(self, device):
        """配置 JAX 设备"""
        if device == "cpu":
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            jax.config.update('jax_platform_name', 'cpu')
            print(f"MAML 设备配置: CPU")
        elif device == "cuda":
            try:
                gpu_devices = jax.devices('gpu')
                if len(gpu_devices) > 0:
                    jax.config.update('jax_platform_name', 'gpu')
                    print(f"MAML 设备配置: GPU - {gpu_devices[0]}")
                else:
                    print("警告: MAML 未检测到 GPU 设备，回退到 CPU")
                    jax.config.update('jax_platform_name', 'cpu')
            except RuntimeError:
                print("警告: MAML GPU 初始化失败，回退到 CPU")
                jax.config.update('jax_platform_name', 'cpu')
        
        print(f"MAML 当前后端: {jax.default_backend()}")
        self.device = device

    def _init_params(self):
        layer_sizes = self.settings['network_architecture']
        key = jax.random.PRNGKey(0)
        params = []
        for i in range(len(layer_sizes) - 1):
            in_size, out_size = layer_sizes, layer_sizes[i+1]
            key, subkey = jax.random.split(key)
            w = jax.random.normal(subkey, (in_size, out_size)) * jnp.sqrt(2. / in_size)
            b = jnp.zeros(out_size)
            params.append((w, b))
        return params

    @staticmethod
    @jax.jit
    def forward(params, x):
        for w, b in params[:-1]:
            x = jax.nn.relu(x @ w + b)
        w_last, b_last = params[-1]
        return x @ w_last + b_last

    @staticmethod
    @jax.jit
    def loss_fn(params, x, y):
        y_pred = MAMLRegressor.forward(params, x)
        return jnp.mean((y - y_pred)**2)

    def inner_loop_update(self, params, support_x, support_y):
        grad_fn = jax.grad(self.loss_fn)
        inner_optimizer = optax.adam(learning_rate=self.inner_lr)
        opt_state = inner_optimizer.init(params)
        
        for _ in range(self.inner_steps):
            grads = grad_fn(params, support_x, support_y)
            updates, opt_state = inner_optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        return params

    def fit(self, tasks, epochs=100):
        @jax.jit
        def meta_update_step(meta_params, single_task):
            support_x, support_y, query_x, query_y = single_task
            
            def query_loss_for_grad(p):
                p_adapted = self.inner_loop_update(p, support_x, support_y)
                return self.loss_fn(p_adapted, query_x, query_y)
            
            meta_grads = jax.grad(query_loss_for_grad)(meta_params)
            return meta_grads

        for epoch in tqdm(range(epochs), desc="元学习训练"):
            meta_gradients_sum = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), self.meta_params)
            for task in tasks:
                meta_grads_task = meta_update_step(self.meta_params, task)
                meta_gradients_sum = jax.tree_util.tree_map(lambda x, y: x + y, meta_gradients_sum, meta_grads_task)
            
            avg_meta_grads = jax.tree_util.tree_map(lambda x: x / len(tasks), meta_gradients_sum)
            updates, self.opt_state = self.optimizer.update(avg_meta_grads, self.opt_state)
            self.meta_params = optax.apply_updates(self.meta_params, updates)

    def predict(self, X, adapted_params=None):
        params_to_use = adapted_params if adapted_params is not None else self.meta_params
        # JAX jit a forward pass for performance
        return jax.jit(self.forward)(params_to_use, X)

    def adapt(self, support_x, support_y):
        print("使用新的支持集样本进行快速适应...")
        adapted_params = self.inner_loop_update(self.meta_params, support_x, support_y)
        return adapted_params