import json

# Load your JSON data
with open("data_save/synthetic_entities/test_72/profiles.json", 'r', encoding='utf-8') as file:
    entities = json.load(file)

# Initialize the result dictionary
common_attributes_dict = {}

# Main loop to find entities with common attributes
for i, entity in enumerate(entities):
    entity_name = entity['name']
    entities_with_common_attrs = []
    
    # Compare current entity with all other entities
    for j, other_entity in enumerate(entities):
        if i != j:  # Skip comparing entity with itself
            # Check if they share any attribute (excluding 'name')
            shares_attribute = False
            
            for attribute_key in entity.keys():
                if attribute_key != 'name':  # Don't match on names
                    if entity[attribute_key] == other_entity[attribute_key]:
                        shares_attribute = True
                        break  # Found at least one match, no need to check more
            
            # If they share an attribute, add to the list
            if shares_attribute:
                entities_with_common_attrs.append(other_entity['name'])
    
    # Store the result in our dictionary
    common_attributes_dict[entity_name] = entities_with_common_attrs

# Example: Print results for a few entities
print("Results:")
for i, (name, related_entities) in enumerate(common_attributes_dict.items()):
    if i < 5:  # Show first 5 examples
        print(f"{name}: {len(related_entities)} entities with common attributes")
        print(f"  First few: {related_entities[:3]}")
        print() 