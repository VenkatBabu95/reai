#!/usr/bin/env python3
"""
Test script to verify blockchain-integrated model training.
"""
import json
import sys

# Import modules
from blockchain import blockchain
from custom_model import custom_llm
from model import (
    train_model_on_batch,
    find_similar_by_embedding,
    get_model_info,
    get_blockchain_stats
)


def test_model_training():
    """Test training the custom LLM and blockchain logging."""
    print("=" * 60)
    print("Testing Custom Medical LLM with Blockchain")
    print("=" * 60)
    
    # Sample medical queries for training
    training_texts = [
        "I have a fever and body aches",
        "My throat is very sore",
        "I'm experiencing chest pain",
        "I have a persistent cough",
        "I'm feeling dizzy and nauseous"
    ]
    
    print("\n1. Training model on medical queries...")
    for i, text in enumerate(training_texts, 1):
        print(f"   Batch {i}: {text}")
        metrics = train_model_on_batch([text], learning_rate=0.01)
        print(f"      → Model hash: {metrics['model_hash_after']}")
    
    # Mine blocks
    print("\n2. Mining blockchain blocks...")
    block_result = blockchain.mine_block(miner_id='test-trainer')
    print(f"   Block {block_result['block_index']} mined!")
    print(f"   Block hash: {block_result['block_hash']}")
    print(f"   Nonce: {block_result['nonce']}")
    
    # Get model info
    print("\n3. Model Information:")
    model_info = get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Get blockchain stats
    print("\n4. Blockchain Statistics:")
    stats = get_blockchain_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test encoding
    print("\n5. Testing embeddings...")
    test_query = "I have a high fever"
    embedding = custom_llm.encode([test_query])
    print(f"   Query: '{test_query}'")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding norm: {np.linalg.norm(embedding):.4f}")
    
    # Test checkpoint save
    print("\n6. Saving model checkpoint...")
    checkpoint_hash = custom_llm.save_checkpoint('checkpoints/model_checkpoint.json')
    print(f"   Checkpoint hash: {checkpoint_hash}")
    block_result = blockchain.mine_block(miner_id='checkpoint-test')
    print(f"   Checkpoint logged to block {block_result['block_index']}")
    
    # Verify blockchain integrity
    print("\n7. Verifying blockchain integrity...")
    is_valid = blockchain.verify_integrity()
    print(f"   Blockchain valid: {is_valid}")
    
    # Get full blockchain history
    print("\n8. Blockchain Training History:")
    history = blockchain.get_model_history('custom-medical-llm-v1')
    print(f"   Total records: {len(history)}")
    for record in history[:5]:  # Show first 5
        print(f"   - Type: {record['type']}, Time: {record['timestamp']}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    import numpy as np
    test_model_training()
