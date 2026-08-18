#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final verification test for VectorHub MVP."""

from vectorhub import VectorCollection
import numpy as np

print("=" * 60)
print("FINAL VERIFICATION TEST")
print("=" * 60)

# Test 1: Basic creation and operations
print("\n[Test 1] Basic Creation and Add")
db = VectorCollection(dim=5, metric="l2")
vectors = [[1.0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
db.add(vectors=vectors, ids=[1, 2, 3])
print("  PASSED: Created collection, added 3 vectors")

# Test 2: Search functionality
print("\n[Test 2] Search Functionality")
results = db.search([1.0, 0, 0, 0, 0], k=2)
assert len(results) == 2
assert results[0][0] == 1  # Closest is vector 1
assert results[0][1] < 0.1  # Distance near 0
print("  PASSED: Search returns correct results")

# Test 3: NumPy array support
print("\n[Test 3] NumPy Array Support")
db2 = VectorCollection(dim=2)
vectors_np = np.array([[1, 0], [0, 1], [1, 1]])
ids_np = np.array([10, 20, 30])
db2.add(vectors=vectors_np, ids=ids_np)
query_np = np.array([1.0, 0.0])
results_np = db2.search(query_np, k=2)
assert len(results_np) == 2
print("  PASSED: NumPy arrays work correctly")

# Test 4: Error handling
print("\n[Test 4] Error Handling")
try:
    db.search([1, 2], k=1)  # Wrong dimension
    assert False, "Should have raised error"
except ValueError as e:
    print("  PASSED: Dimension mismatch detected")

try:
    db.add(vectors=[[1, 0, 0, 0, 0]], ids=[1])  # Duplicate ID
    assert False, "Should have raised error"
except RuntimeError as e:
    print("  PASSED: Duplicate ID detected")

# Test 5: Edge cases
print("\n[Test 5] Edge Cases")
db3 = VectorCollection(dim=3)
db3.add(vectors=[[1, 2, 3]], ids=[100])
results = db3.search([1, 2, 3], k=100)  # k > dataset size
assert len(results) == 1
print("  PASSED: k > dataset size handled correctly")

print("\n" + "=" * 60)
print("ALL VERIFICATION TESTS PASSED!")
print("=" * 60)
