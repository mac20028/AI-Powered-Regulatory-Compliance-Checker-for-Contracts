import os

# Folder containing TXT files
contracts_folder = "full_contract_txt"  

# List all TXT files in the folder
contract_files = [f for f in os.listdir(contracts_folder) if f.endswith(".txt")]

# Read and print each contract
for file in contract_files:
    file_path = os.path.join(contracts_folder, file)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"\n=== {file} ===")
    print(text[:1000])  
    print("\n=== End of Preview ===")
