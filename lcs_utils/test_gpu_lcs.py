import torch
import time
from gpu_alignment import parallel_step_lcs_score

def test_correctness():
    print("\n>>> 正在进行正确性测试...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 构造 Case
    # A (标准): [1, 1, 2, 2] (Step0: 1,1; Step1: 2,2)
    # B1 (完美): [1, 1, 2, 2] -> Score 应为 1.0
    # B2 (乱序): [2, 2, 1, 1] -> Score 应较低 (无法同时匹配Step0和Step1)
    
    maj_ids = torch.tensor([[1, 1, 2, 2]]).to(device)
    # 熵：第2个位置(index 1)高熵，切分 -> Step0: indices[0,1], Step1: indices[2,3]
    maj_ents = torch.tensor([[0.1, 5.0, 0.1, 0.1]]).to(device)
    
    stud_ids = torch.tensor([
        [
            [1, 1, 2, 2], # Sample 0: Perfect
            [2, 2, 1, 1]  # Sample 1: Reversed
        ]
    ]).to(device)
    
    scores = parallel_step_lcs_score(maj_ids, maj_ents, stud_ids, entropy_threshold=2.0)
    
    print(f"标准序列: {maj_ids.cpu().tolist()}")
    print(f"学生序列1: {stud_ids[0,0].cpu().tolist()} -> 分数: {scores[0,0]:.4f}")
    print(f"学生序列2: {stud_ids[0,1].cpu().tolist()} -> 分数: {scores[0,1]:.4f}")
    
    if scores[0,0] > scores[0,1]:
        print("✅ 测试通过：顺序敏感性验证成功 (正序分 > 逆序分)")
    else:
        print("❌ 测试失败：算法未能捕捉顺序差异")

def benchmark_speed():
    print("\n>>> 正在进行极限性能压测...")
    if not torch.cuda.is_available():
        print("⚠️ 未检测到 GPU，跳过性能测试")
        return

    # 模拟真实 TTRL 训练规模
    # Batch=8, Sample=32, Length=2048 (较长文本)
    B, K, L = 8, 32, 2048
    device = "cuda"
    
    print(f"数据规模: Batch={B}, Sample={K}, Len={L}")
    print(f"总计算量: {B*K} 对序列，每对长度 {L}")
    
    # 随机生成数据
    maj_ids = torch.randint(1, 10000, (B, L), device=device)
    maj_ents = torch.rand((B, L), device=device) * 5.0
    stud_ids = torch.randint(1, 10000, (B, K, L), device=device)
    
    # 预热 CUDA
    for _ in range(5):
        _ = parallel_step_lcs_score(maj_ids, maj_ents, stud_ids)
    torch.cuda.synchronize()
    
    # 计时
    start = time.time()
    iters = 100
    for _ in range(iters):
        _ = parallel_step_lcs_score(maj_ids, maj_ents, stud_ids)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iters
    print(f"平均耗时: {avg_time*1000:.2f} ms")
    print(f"FPS (Samples/sec): {(B*K)/avg_time:.0f}")
    
    if avg_time < 0.05: # 50ms
        print("🚀 性能评价：极速 (适合在线训练)")
    else:
        print("⚠️ 性能评价：一般")

if __name__ == "__main__":
    test_correctness()
    benchmark_speed()