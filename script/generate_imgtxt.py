from pathlib import Path

# Define the directory to traverse and the output file
directory_to_traverse = Path("build")
output_file = Path("build/output.txt")

# Open the output file for writing
with output_file.open("w") as file:
    # Traverse the directory
    for jpg_file in directory_to_traverse.rglob("*.jpg"):
        # Write the absolute path of each jpg file to the output file
        file.write(str(jpg_file.resolve()) + "\n")

print(f"All .jpg file paths have been written to {output_file}")
