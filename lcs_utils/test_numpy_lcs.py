import numpy as np
import time
from gpu_alignment import numpy_step_lcs_score

def test_correctness():
    print("\n>>> 正在进行正确性测试 (CPU/Numpy)...")
    
    # 构造数据: Batch=1, Sample=2, L_maj=4, L_stu=4
    # 众数: [1, 1, 2, 2] -> 熵切分 -> Step0:[1,1], Step1:[2,2]
    maj_ids = np.array([[1, 1, 2, 2]])
    maj_ents = np.array([[0.1, 5.0, 0.1, 0.1]]) # index 1 处切分
    
    stud_ids = np.array([
        [
            [1, 1, 2, 2], # 正序 (应该高分)
            [2, 2, 1, 1]  # 逆序 (应该低分)
        ]
    ])
    
    scores = numpy_step_lcs_score(maj_ids, maj_ents, stud_ids, entropy_threshold=2.0)
    
    print(f"标准序列: {maj_ids.tolist()}")
    print(f"学生1 (正序): {stud_ids[0,0].tolist()} -> 分数: {scores[0,0]:.4f}")
    print(f"学生2 (逆序): {stud_ids[0,1].tolist()} -> 分数: {scores[0,1]:.4f}")
    
    if scores[0,0] > scores[0,1]:
        print("✅ 测试通过：顺序敏感性验证成功 (正序分 > 逆序分)")
    else:
        print("❌ 测试失败：算法未能捕捉顺序差异")

def benchmark_cpu_speed():
    print("\n>>> 正在进行 CPU 性能压测...")
    
    # 模拟规模
    B = 8
    K = 32
    L = 2048  # 长文本
    
    print(f"数据规模: Batch={B}, Sample={K}, Len={L}")
    print("注意：这是纯 CPU 计算，请不要与 GPU 直接比速度")
    
    # 随机数据
    np.random.seed(42)
    maj_ids = np.random.randint(1, 1000, (B, L))
    maj_ents = np.random.rand(B, L) * 5.0
    stud_ids = np.random.randint(1, 1000, (B, K, L))
    
    # 预热
    _ = numpy_step_lcs_score(maj_ids, maj_ents, stud_ids)
    
    start = time.time()
    iters = 5
    for _ in range(iters):
        _ = numpy_step_lcs_score(maj_ids, maj_ents, stud_ids)
    end = time.time()
    
    avg_time = (end - start) / iters
    print(f"平均耗时: {avg_time:.4f} s")
    print(f"单样本耗时: {(avg_time / (B*K))*1000:.2f} ms")
    
    if avg_time < 2.0:
        print("🚀 CPU 性能极佳 (对于Python来说)")
    else:
        print("⚠️ 耗时较高，生产环境建议控制 Sequence Length 或使用 GPU 版")

if __name__ == "__main__":
    test_correctness()
    benchmark_cpu_speed()