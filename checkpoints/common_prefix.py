import json

def get_prefixes(path):
    """Return all prefixes for a given path, excluding the last directory."""
    parts = path.strip('/').split('/')
    prefixes = []
    # Exclude the last part (directory)
    for i in range(4, len(parts)):
        prefixes.append('/'.join(parts[:i]))
    return prefixes

jsonl_file = "weka_paths.jsonl"
all_prefixes = set()

with open(jsonl_file, 'r') as f:
    for line in f:
        obj = json.loads(line)
        weka_path = obj.get("checkpoints_location", "")
        # Remove weka:// or s3:// prefix
        if weka_path.startswith("weka://"):
            weka_path = weka_path[len("weka://"):]
        elif weka_path.startswith("s3://"):
            weka_path = weka_path[len("s3://"):]
        # Get all prefixes for this path, excluding last dir
        for prefix in get_prefixes(weka_path):
            all_prefixes.add(prefix)

# Print all unique prefixes, sorted
for prefix in sorted(all_prefixes):
    print(prefix)