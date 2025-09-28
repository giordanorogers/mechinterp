import json

def find_entities_with_common_attributes(json_file_path):
    """
    Creates a dictionary where keys are entity names and values are lists of other entities
    that share at least one common attribute with the key entity.
    """
    
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as file:
        entities = json.load(file)
    
    # Dictionary to store the results
    common_attributes_dict = {}
    
    # Get all attribute keys (excluding 'name' since we don't want to match on names)
    if entities:
        attribute_keys = [key for key in entities[0].keys() if key != 'name']
    
    # For each entity, find others with common attributes
    for i, entity in enumerate(entities):
        entity_name = entity['name']
        entities_with_common_attrs = []
        
        # Compare with all other entities
        for j, other_entity in enumerate(entities):
            if i != j:  # Don't compare entity with itself
                other_name = other_entity['name']
                
                # Check if they share any common attribute values
                has_common_attribute = False
                for attr_key in attribute_keys:
                    if entity.get(attr_key) == other_entity.get(attr_key):
                        has_common_attribute = True
                        break
                
                # If they share at least one attribute, add to the list
                if has_common_attribute:
                    entities_with_common_attrs.append(other_name)
        
        # Store the result
        common_attributes_dict[entity_name] = entities_with_common_attrs
    
    return common_attributes_dict

def find_specific_common_attributes(json_file_path):
    """
    Enhanced version that also shows which specific attributes are shared.
    Returns a dictionary where values are dictionaries containing shared attributes.
    """
    
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as file:
        entities = json.load(file)
    
    # Dictionary to store the results
    detailed_common_attributes = {}
    
    # Get all attribute keys (excluding 'name')
    if entities:
        attribute_keys = [key for key in entities[0].keys() if key != 'name']
    
    # For each entity, find others with common attributes
    for i, entity in enumerate(entities):
        entity_name = entity['name']
        detailed_common_attributes[entity_name] = {}
        
        # Compare with all other entities
        for j, other_entity in enumerate(entities):
            if i != j:  # Don't compare entity with itself
                other_name = other_entity['name']
                shared_attributes = {}
                
                # Find all shared attribute values
                for attr_key in attribute_keys:
                    if entity.get(attr_key) == other_entity.get(attr_key):
                        shared_attributes[attr_key] = entity.get(attr_key)
                
                # If they share at least one attribute, add to the result
                if shared_attributes:
                    detailed_common_attributes[entity_name][other_name] = shared_attributes
    
    return detailed_common_attributes

# Example usage
if __name__ == "__main__":
    # Path to your JSON file
    json_file_path = "data_save/synthetic_entities/test_72/profiles.json"
    
    # Get simple dictionary of entities with common attributes
    print("=== Simple Common Attributes Dictionary ===")
    simple_result = find_entities_with_common_attributes(json_file_path)
    
    # Print first few examples
    for i, (entity, related_entities) in enumerate(simple_result.items()):
        if i < 3:  # Show first 3 examples
            print(f"\n{entity} shares attributes with:")
            for related in related_entities[:5]:  # Show first 5 related entities
                print(f"  - {related}")
            if len(related_entities) > 5:
                print(f"  ... and {len(related_entities) - 5} more")
    
    print(f"\nTotal entities processed: {len(simple_result)}")
    
    # Get detailed dictionary showing which specific attributes are shared
    print("\n=== Detailed Common Attributes (showing specific shared attributes) ===")
    detailed_result = find_specific_common_attributes(json_file_path)
    
    # Print a detailed example for the first entity
    first_entity = list(detailed_result.keys())[0]
    print(f"\nDetailed breakdown for {first_entity}:")
    for related_entity, shared_attrs in list(detailed_result[first_entity].items())[:3]:
        print(f"  Shares with {related_entity}:")
        for attr, value in shared_attrs.items():
            print(f"    - {attr}: {value}")
        print() 