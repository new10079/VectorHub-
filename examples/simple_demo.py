#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple demo of VectorHub functionality."""

from vectorhub import VectorCollection

print("VectorHub MVP Demo")
print("=" * 50)

# Create a collection
db = VectorCollection(dim=3, metric="l2")
print("\n[OK] Created VectorCollection with dim=3")

# Add some vectors
vectors = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.5, 0.5, 0.5],
]
ids = [101, 102, 103, 104, 105]

db.add(vectors=vectors, ids=ids)
print("[OK] Added %d vectors with IDs: %s" % (len(vectors), ids))

# Search for nearest neighbors
query = [1.0, 0.1, 0.0]
k = 2
results = db.search(query, k=k)

print("\n[OK] Search query: %s, k=%d" % (query, k))
print("Results (id, distance):")
for id_, dist in results:
    print("  ID: %3d, Distance: %.6f" % (id_, dist))

# Test with another query
print("\n" + "=" * 50)
query2 = [0.0, 0.0, 1.0]
results2 = db.search(query2, k=3)
print("\nSearch query: %s, k=3" % (query2,))
print("Results (id, distance):")
for id_, dist in results2:
    print("  ID: %3d, Distance: %.6f" % (id_, dist))

print("\n[OK] Demo completed successfully!")

