import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import time
from gpu_alignment import fast_parallel_step_lcs_score

def get_memory_usage():
    """获取当前 GPU 显存占用（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0

def get_memory_reserved():
    """获取当前 GPU 保留的显存（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024**2
    return 0.0

def get_max_memory_usage():
    """获取峰值 GPU 显存占用（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0

def reset_memory_stats():
    """重置显存统计"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        # 强制同步，确保重置生效
        torch.cuda.synchronize()

# 引入原来的版本做对比（假设保存为 gpu_alignment.py）
try:
    from gpu_alignment import parallel_step_lcs_score as baseline_func
    HAS_BASELINE = True
except ImportError:
    HAS_BASELINE = False

def benchmark():
    if not torch.cuda.is_available():
        return

    # 极端负载
    B, L_maj, L_stu = 8, 3072, 3072
    device = "cuda"
    
    print(f"Testing Scale: B={B}, L_maj={L_maj}, L_stu={L_stu}")
    print("-" * 40)
    
    maj_ids = torch.randint(1, 5000, (B, L_maj), device=device)
    maj_ents = torch.rand((B, L_maj), device=device) * 5.0
    stud_ids = torch.randint(1, 5000, (B, L_stu), device=device)
    
    # 1. 原始版本测试（如果存在）
    if HAS_BASELINE:
        # 预热
        _ = baseline_func(maj_ids, maj_ents, stud_ids)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(20):
            _ = baseline_func(maj_ids, maj_ents, stud_ids)
        torch.cuda.synchronize()
        base_time = (time.time() - start) / 20
        print(f"Original (Einsum+Bool): {base_time*1000:.2f} ms")
    
    # 2. 优化版本测试
    # 预热
    _ = fast_parallel_step_lcs_score(maj_ids, maj_ents, stud_ids)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(32):
        _ = fast_parallel_step_lcs_score(maj_ids, maj_ents, stud_ids)
    torch.cuda.synchronize()
    fast_time = (time.time() - start) / 32
    print(f"Optimized (BMM+FP16):   {fast_time*1000:.2f} ms")
    
    if HAS_BASELINE:
        print(f"Speedup: {base_time / fast_time:.2f}x")

def test_memory_usage():
    """测试 GPU 显存占用"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test.")
        return
    
    print("\n" + "=" * 60)
    print("GPU Memory Usage Test")
    print("=" * 60)
    
    # 重置显存统计
    reset_memory_stats()
    
    # 记录初始显存
    initial_memory = get_memory_usage()
    print(f"Initial Memory: {initial_memory:.2f} MB")
    
    # 测试规模
    B, L_maj, L_stu = 32, 3072, 3072
    device = "cuda"
    
    print(f"\nTest Scale: B={B}, L_maj={L_maj}, L_stu={L_stu}")
    print("-" * 60)
    
    # 分配输入数据
    torch.cuda.empty_cache()
    before_data_memory = get_memory_usage()
    
    maj_ids = torch.randint(1, 5000, (B, L_maj), device=device)
    maj_ents = torch.rand((B, L_maj), device=device) * 5.0
    stud_ids = torch.randint(1, 5000, (B, L_stu), device=device)
    torch.cuda.synchronize()
    
    after_data_memory = get_memory_usage()
    data_memory = after_data_memory - before_data_memory
    print(f"Input Data Memory: {data_memory:.2f} MB")
    print(f"  - maj_ids: {maj_ids.element_size() * maj_ids.numel() / 1024**2:.2f} MB")
    print(f"  - maj_ents: {maj_ents.element_size() * maj_ents.numel() / 1024**2:.2f} MB")
    print(f"  - stud_ids: {stud_ids.element_size() * stud_ids.numel() / 1024**2:.2f} MB")
    
    # 测试原始版本（如果存在）
    baseline_peak_overhead = None
    if HAS_BASELINE:
        print(f"\n{'='*60}")
        print("Testing Original (Einsum+Bool) Version")
        print(f"{'='*60}")
        
        # 完全清理并重置
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        reset_memory_stats()
        torch.cuda.synchronize()
        
        before_func_allocated = get_memory_usage()
        before_func_reserved = get_memory_reserved()
        
        # 执行函数
        result = baseline_func(maj_ids, maj_ents, stud_ids)
        torch.cuda.synchronize()
        
        # 立即测量峰值（在释放结果之前）
        peak_allocated = get_max_memory_usage()
        
        after_func_allocated = get_memory_usage()
        after_func_reserved = get_memory_reserved()
        
        # 计算显存增量（这才是函数实际使用的显存）
        func_allocated_overhead = after_func_allocated - before_func_allocated
        func_reserved_overhead = after_func_reserved - before_func_reserved
        baseline_peak_overhead = peak_allocated - before_func_allocated
        
        print(f"\n📊 显存使用情况:")
        print(f"  函数执行前: {before_func_allocated:.2f} MB")
        print(f"  函数执行中峰值: {peak_allocated:.2f} MB ({peak_allocated/1024:.2f} GB)")
        print(f"  函数执行后: {after_func_allocated:.2f} MB")
        print(f"\n✅ 函数实际使用的显存: {baseline_peak_overhead:.2f} MB ({baseline_peak_overhead/1024:.2f} GB)")
        
        # 清理结果
        del result
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 测试优化版本
    print(f"\n{'='*60}")
    print("Testing Optimized (BMM+FP16) Version")
    print(f"{'='*60}")
    
    # 完全清理并重置
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    reset_memory_stats()
    torch.cuda.synchronize()
    
    before_func_allocated = get_memory_usage()
    before_func_reserved = get_memory_reserved()
    
    # 执行函数
    result = fast_parallel_step_lcs_score(maj_ids, maj_ents, stud_ids)
    torch.cuda.synchronize()
    
    # 立即测量峰值（在释放结果之前）
    peak_allocated = get_max_memory_usage()
    
    after_func_allocated = get_memory_usage()
    after_func_reserved = get_memory_reserved()
    
    # 计算显存增量（这才是函数实际使用的显存）
    func_allocated_overhead = after_func_allocated - before_func_allocated
    func_reserved_overhead = after_func_reserved - before_func_reserved
    optimized_peak_overhead = peak_allocated - before_func_allocated
    
    print(f"\n📊 显存使用情况:")
    print(f"  函数执行前: {before_func_allocated:.2f} MB")
    print(f"  函数执行中峰值: {peak_allocated:.2f} MB ({peak_allocated/1024:.2f} GB)")
    print(f"  函数执行后: {after_func_allocated:.2f} MB")
    print(f"\n✅ 函数实际使用的显存: {optimized_peak_overhead:.2f} MB ({optimized_peak_overhead/1024:.2f} GB)")
    
    # 清理结果
    del result
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # 清理所有数据
    del maj_ids, maj_ents, stud_ids
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    final_memory = get_memory_usage()
    final_reserved = get_memory_reserved()
    final_memory = get_memory_usage()
    print(f"\n{'='*60}")
    print("📋 总结")
    print("=" * 60)
    print(f"测试规模: B={B}, L_maj={L_maj}, L_stu={L_stu}")
    if HAS_BASELINE and baseline_peak_overhead is not None:
        print(f"\n原始版本 (parallel_step_lcs_score):")
        print(f"  ✅ 峰值显存: {baseline_peak_overhead:.2f} MB ({baseline_peak_overhead/1024:.2f} GB)")
    print(f"\n优化版本 (fast_parallel_step_lcs_score):")
    print(f"  ✅ 峰值显存: {optimized_peak_overhead:.2f} MB ({optimized_peak_overhead/1024:.2f} GB)")
    print(f"\n💡 说明:")
    print(f"  - 这是函数执行过程中的峰值显存使用量")
    print(f"  - 函数执行完成后，大部分显存会被释放")
    print(f"  - 如果显存不足，可以减小 B、L_maj 或 L_stu 的值")
    print("=" * 60)

def test_correctness():
    # 简单的正确性回归测试，确保优化没有破坏逻辑
    device = "cuda"
    maj_ids = torch.tensor([[1, 2, 3]]).to(device)  # [1, 3]
    maj_ents = torch.tensor([[0.1, 9.0, 0.1]]).to(device)  # Step0: 1, Step1: 2,3
    stud_ids_1 = torch.tensor([[1, 2, 3]]).to(device)  # [1, 3] - 完全匹配
    stud_ids_2 = torch.tensor([[3, 2, 1]]).to(device)  # [1, 3] - 部分匹配
    
    s1 = fast_parallel_step_lcs_score(maj_ids, maj_ents, stud_ids_1, entropy_threshold=1.0)
    s2 = fast_parallel_step_lcs_score(maj_ids, maj_ents, stud_ids_2, entropy_threshold=1.0)
    print(f"Score for [1,2,3]: {s1.item():.4f}")
    print(f"Score for [3,2,1]: {s2.item():.4f}")
    assert s1[0] > s2[0], f"Expected s1 ({s1[0]}) > s2 ({s2[0]})"
    print("✅ Correctness Check Passed.")

if __name__ == "__main__":
    test_correctness()
    benchmark()
    test_memory_usage()