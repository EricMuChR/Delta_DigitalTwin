import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import cma
import optax
from scipy.optimize import minimize
import os

def configure_jax_device(device):
    """配置 JAX 使用的设备"""
    if device == "cpu":
        # 强制 JAX 使用 CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        jax.config.update('jax_platform_name', 'cpu')
        print("JAX 设备配置: CPU")
    elif device == "cuda":
        # 首先检查是否安装了 GPU 版本的 JAX
        try:
            gpu_devices = jax.devices('gpu')
            if len(gpu_devices) > 0:
                jax.config.update('jax_platform_name', 'gpu')
                print(f"JAX 设备配置: GPU - {gpu_devices[0]}")
                print(f"可用的 GPU 设备: {gpu_devices}")
            else:
                print("警告: 未检测到 GPU 设备")
                print("可能原因:")
                print("  1. 未安装 jax[cuda] 版本，请运行: pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
                print("  2. CUDA 驱动未正确安装")
                print("  3. GPU 不可用或被其他程序占用")
                print("回退到 CPU 模式")
                jax.config.update('jax_platform_name', 'cpu')
        except RuntimeError as e:
            print(f"GPU 初始化失败: {e}")
            print("这通常意味着您安装的是 CPU 版本的 JAX")
            print("要使用 GPU，请安装: pip install --upgrade jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
            print("回退到 CPU 模式")
            jax.config.update('jax_platform_name', 'cpu')
    else:
        print(f"未知设备类型: {device}，使用默认配置")
    
    # 打印当前实际使用的设备
    print(f"当前 JAX 后端: {jax.default_backend()}")
    print(f"可用设备列表: {jax.devices()}")

def get_loss_function(robot, cmd_pos_jax, meas_pos_jax):
    meas_error = meas_pos_jax - cmd_pos_jax
    
    @jax.jit
    def calculate_total_deviation(delta_p):
        thetas_nominal = jax.vmap(robot.inverse_kinematics)(cmd_pos_jax)
        sim_pos_actual = jax.vmap(robot.forward_kinematics_with_errors, in_axes=(0, None))(thetas_nominal, delta_p)
        sim_error = sim_pos_actual - cmd_pos_jax
        recognition_deviations = jnp.linalg.norm(sim_error - meas_error, axis=1)
        return jnp.sum(recognition_deviations)
        
    return calculate_total_deviation

def global_search_cmaes(loss_fn, settings):
    print("--- 阶段1a: 全局探索 (CMA-ES) ---")
    ga_settings = settings['global_search']
    initial_guess = np.zeros(42)
    sigma0 = (ga_settings['param_upper_bound'] - ga_settings['param_lower_bound']) / 5.0
    
    options = {'bounds': [ga_settings['param_lower_bound'], ga_settings['param_upper_bound']],
               'maxiter': ga_settings['generations'],
               'popsize': ga_settings['population_size']}
               
    es = cma.CMAEvolutionStrategy(initial_guess, sigma0, options)
    
    pbar = tqdm(total=ga_settings['generations'], desc="CMA-ES 全局探索")
    for gen in range(ga_settings['generations']):
        solutions = es.ask()
        fitness_values = [float(loss_fn(jnp.array(x))) for x in solutions]
        es.tell(solutions, fitness_values)
        pbar.update(1)
        pbar.set_postfix({'Best': f"{es.result.fbest:.6f}"})
    pbar.close()
    
    return jnp.array(es.result.xbest)

def local_coarse_adam(loss_fn, initial_params, settings):
    print("--- 阶段1b: 局部粗调 (Adam) ---")
    adam_settings = settings['local_finetune']['adam']
    params = initial_params
    
    optimizer = optax.adam(learning_rate=adam_settings['learning_rate'])
    opt_state = optimizer.init(params)
    grad_fn = jax.grad(loss_fn)

    pbar = tqdm(range(adam_settings['max_iterations']), desc="Adam 局部粗调")
    for i in pbar:
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        if i % 10 == 0:
            loss_val = loss_fn(params)
            pbar.set_description(f"Adam 迭代 {i+1} | 损失: {loss_val:.6f}")
            if jnp.linalg.norm(grads) < adam_settings['convergence_grad_norm']:
                print(f"\nAdam 在第 {i+1} 次迭代收敛。")
                break
                
    print(f"Adam 粗调完成，最终损失: {loss_fn(params):.6f}")
    return params

def local_fine_lbfgs(loss_fn, initial_params, settings):
    print("--- 阶段1c: 局部精调 (L-BFGS-B), 此过程可能需要几分钟且无进度条... ---")
    lbfgs_settings = settings['local_finetune']['lbfgs']
    
    @jax.jit
    def loss_and_grad(p):
        return loss_fn(p), jax.grad(loss_fn)(p)

    def scipy_obj_func(p_np):
        p_jnp = jnp.array(p_np)
        loss, grad = loss_and_grad(p_jnp)
        return np.array(loss, dtype=np.float64), np.array(grad, dtype=np.float64)

    result = minimize(scipy_obj_func, 
                      x0=np.array(initial_params), 
                      method='L-BFGS-B', 
                      jac=True, 
                      options={'maxiter': lbfgs_settings['max_iterations'], 
                               'gtol': lbfgs_settings['convergence_gtol'],
                               'disp': True})

    final_params = jnp.array(result.x)
    print(f"L-BFGS-B 精调完成，最终损失: {result.fun:.6f}")
    return final_params

def identify_parameters(robot, cmd_pos, meas_pos, settings, device='cpu'):
    # 配置设备
    configure_jax_device(device)
    
    cmd_pos_jax = jnp.array(cmd_pos)
    meas_pos_jax = jnp.array(meas_pos)
    loss_fn = get_loss_function(robot, cmd_pos_jax, meas_pos_jax)
    
    params_after_global = global_search_cmaes(loss_fn, settings)
    params_after_coarse = local_coarse_adam(loss_fn, params_after_global, settings)
    final_params = local_fine_lbfgs(loss_fn, params_after_coarse, settings)
    
    return np.array(final_params)