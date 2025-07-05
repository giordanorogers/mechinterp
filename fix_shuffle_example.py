import random

# Example data
additional_entities = ["Entity A", "Entity B", "Entity C"]
target_entity = ["New Entity"]

print("Original additional_entities:", additional_entities)
print("Target entity to add:", target_entity)

# PROBLEM: This doesn't work because extend() and shuffle() return None
# print(random.shuffle(additional_entities.extend(target_entity)))

# SOLUTION 1: Do operations separately
additional_entities.extend(target_entity)  # Add target_entity to the list
random.shuffle(additional_entities)       # Shuffle the list in place
print("Solution 1 - After extend and shuffle:", additional_entities)

# SOLUTION 2: Create a new combined list, then shuffle
additional_entities_2 = ["Entity A", "Entity B", "Entity C"]
target_entity_2 = ["New Entity"]

combined_list = additional_entities_2 + target_entity_2  # Create new list
random.shuffle(combined_list)  # Shuffle the new list
print("Solution 2 - Combined and shuffled:", combined_list)

# SOLUTION 3: If target_entity is a single item (not a list)
additional_entities_3 = ["Entity A", "Entity B", "Entity C"]
single_target = "Single New Entity"

additional_entities_3.append(single_target)  # Add single item
random.shuffle(additional_entities_3)        # Shuffle
print("Solution 3 - Append single item and shuffle:", additional_entities_3)

# SOLUTION 4: One-liner using random.sample() for a shuffled copy
additional_entities_4 = ["Entity A", "Entity B", "Entity C"]
target_entity_4 = ["New Entity"]

shuffled_copy = random.sample(additional_entities_4 + target_entity_4, 
                             len(additional_entities_4 + target_entity_4))
print("Solution 4 - Shuffled copy using random.sample:", shuffled_copy) 