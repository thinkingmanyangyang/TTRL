#!/usr/bin/env python3
"""
步骤1: 只进行推理，保存原始结果到缓存
不进行 LCS 分析，只调用 API 获取回答并保存
"""

import asyncio
import aiohttp
import time
import sys
import argparse
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict

# 数据配置
DATA_PATH = "/caobing/biomedical/TTRL/verl/data/AIME-TTT/test.parquet"
API_BASE_URL = "http://localhost:2333/v1"
MODEL_NAME = "math-model"

# 推理参数默认值
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOKENS = 3072
N_SAMPLES = 32


async def call_api_single(session: aiohttp.ClientSession, question: str, index: int, 
                          n_samples: int, temperature: float, top_p: float, max_tokens: int) -> Dict:
    """异步调用 API 单个请求"""
    
    url = f"{API_BASE_URL}/completions"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": f"User: {question}\n\nAssistant:",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n_samples,
    }
    
    try:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            choices = result.get("choices", [])
            responses = [choice.get("text", "").strip() for choice in choices]
            
            return {
                "index": index,
                "question": question,
                "responses": responses,
                "success": True
            }
    except Exception as e:
        return {
            "index": index,
            "question": question,
            "responses": [],
            "success": False,
            "error": str(e)
        }


async def batch_inference(questions: List[str], args) -> List[Dict]:
    """批量并发推理"""
    
    print(f"开始批量推理 {len(questions)} 个问题...")
    print(f"并发数: {args.max_concurrent}, 每题生成 {args.n_samples} 个答案\n")
    
    semaphore = asyncio.Semaphore(args.max_concurrent)
    
    async def limited_call(session, question, index):
        async with semaphore:
            return await call_api_single(session, question, index,
                                        args.n_samples, args.temperature, 
                                        args.top_p, args.max_tokens)
    
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [limited_call(session, question, i) for i, question in enumerate(questions)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        print(f"✓ 批量推理完成！耗时: {elapsed:.2f}s")
        print(f"  平均速度: {elapsed/len(questions):.2f}s/问题\n")
        
        return results


def load_questions_from_dataset(data_path: str, start_index: int, end_index: int, num_samples: int) -> List[str]:
    """从数据集中加载测试问题"""
    
    print(f"从数据集加载问题: {data_path}")
    
    if not Path(data_path).exists():
        print(f"❌ 数据集文件不存在: {data_path}")
        return []
    
    df = pd.read_parquet(data_path)
    total_size = len(df)
    print(f"✓ 数据集加载成功，共 {total_size} 条数据")
    
    if num_samples is not None:
        actual_end = min(start_index + num_samples, total_size)
    elif end_index is not None:
        actual_end = min(end_index, total_size)
    else:
        actual_end = total_size
    
    actual_start = min(start_index, total_size)
    print(f"  读取范围: [{actual_start}, {actual_end}) (共 {actual_end - actual_start} 条)")
    
    question_col = 'prompt' if 'prompt' in df.columns else df.columns[0]
    
    questions = []
    for i, row in df.iloc[actual_start:actual_end].iterrows():
        question = row[question_col]
        if isinstance(question, list):
            for msg in reversed(question):
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    question = msg.get('content', '')
                    break
        questions.append(str(question))
    
    print(f"✓ 已加载 {len(questions)} 条问题\n")
    return questions


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="vLLM API 批量推理（仅推理，不分析）")
    
    # 数据配置
    parser.add_argument("--start-index", type=int, default=0, help="起始索引")
    parser.add_argument("--end-index", type=int, default=30, help="结束索引")
    parser.add_argument("--num-samples", type=int, default=None, help="读取数量")
    
    # 推理配置
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES, help="每题生成答案数")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="采样温度")
    parser.add_argument("--top-p", type=float, default=TOP_P, help="Top-p")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="最大 tokens")
    parser.add_argument("--max-concurrent", type=int, default=20, help="最大并发数")
    
    # 缓存配置
    parser.add_argument("--cache-dir", type=str, default="inference_cache", help="推理结果保存目录")
    parser.add_argument("--cache-name", type=str, default="responses", help="缓存文件名（不含扩展名）")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载问题
    questions = load_questions_from_dataset(DATA_PATH, args.start_index, args.end_index, args.num_samples)
    
    if not questions:
        print("没有问题需要推理")
        return
    
    print("="*80)
    print("vLLM API 批量推理（仅推理模式）")
    print("="*80)
    print(f"API: {API_BASE_URL}")
    print(f"问题数: {len(questions)}")
    print(f"每题样本数: {args.n_samples}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print(f"结果保存目录: {args.cache_dir}")
    print("="*80 + "\n")
    
    # 检查 API
    import requests
    try:
        response = requests.get(f"{API_BASE_URL.replace('/v1', '')}/health", timeout=5)
        if response.status_code != 200:
            print("❌ vLLM API 服务未启动！")
            return
    except Exception as e:
        print(f"❌ 无法连接到 vLLM API: {e}")
        return
    
    print("✓ API 服务连接成功\n")
    
    # 运行推理
    results = asyncio.run(batch_inference(questions, args))
    
    # 保存结果到缓存
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"{args.cache_name}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'results': results,
            'metadata': {
                'n_samples': args.n_samples,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'max_tokens': args.max_tokens,
                'timestamp': time.time()
            }
        }, f)
    
    print(f"\n✓ 推理结果已保存到: {cache_file}")
    print(f"   缓存大小: {cache_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    success_count = sum(1 for r in results if r['success'])
    print(f"   成功: {success_count}/{len(results)}")
    
    print("\n" + "="*80)
    print("下一步: 运行 analyze_cached.py 进行 LCS 分析")
    print("="*80)
    print(f"示例命令:")
    print(f"  python analyze_cached.py --cache-file {cache_file} --lcs-max-tokens 2048 --lcs-normalize-mode avg")
    print(f"或批量分析:")
    print(f"  bash run_all_analysis.sh")


if __name__ == "__main__":
    main()

