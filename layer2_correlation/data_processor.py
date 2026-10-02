import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def synthesize_xdr_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesizes XDR endpoint telemetry (username, process, file, privilege_level, event_type)
    based on the attack_cat column.
    """
    # Define conditions based on attack_cat
    # These map typical attacks to endpoint telemetry
    conditions = [
        df['attack_cat'] == 'Backdoors',
        df['attack_cat'] == 'DoS',
        df['attack_cat'] == 'Exploits',
        df['attack_cat'] == 'Fuzzers',
        df['attack_cat'] == 'Generic',
        df['attack_cat'] == 'Reconnaissance',
        df['attack_cat'] == 'Shellcode',
        df['attack_cat'] == 'Worms',
        df['attack_cat'] == 'Analysis',
        df['attack_cat'] == 'Normal'
    ]

    # Process injection
    process_choices = [
        'mimikatz.exe',         # Backdoors
        'loic.exe',             # DoS
        'powershell.exe',       # Exploits
        'fuzzer.exe',           # Fuzzers
        'cmd.exe',              # Generic
        'nmap.exe',             # Reconnaissance
        'shell.exe',            # Shellcode
        'worm.exe',             # Worms
        'wireshark.exe',        # Analysis
        'svchost.exe'           # Normal
    ]
    df['process'] = np.select(conditions, process_choices, default='unknown.exe')

    # File injection
    file_choices = [
        'C:\\Windows\\System32\\lsass.dmp', # Backdoors
        'C:\\temp\\dos_log.txt',            # DoS
        'C:\\Windows\\Temp\\exploit.ps1',   # Exploits
        'C:\\temp\\crash.dmp',              # Fuzzers
        'C:\\temp\\generic.log',            # Generic
        'C:\\temp\\scan.txt',               # Reconnaissance
        'C:\\temp\\shell.bin',              # Shellcode
        'C:\\Windows\\smb.vbs',             # Worms
        'C:\\temp\\pcap.cap',               # Analysis
        'C:\\Windows\\System32\\hal.dll'    # Normal
    ]
    df['file'] = np.select(conditions, file_choices, default='none')

    # Username injection
    user_choices = [
        'SYSTEM',               # Backdoors
        'NETWORK_SERVICE',      # DoS
        'admin',                # Exploits
        'guest',                # Fuzzers
        'user',                 # Generic
        'anonymous',            # Reconnaissance
        'web_app',              # Shellcode
        'SYSTEM',               # Worms
        'admin',                # Analysis
        'SYSTEM'                # Normal
    ]
    df['username'] = np.select(conditions, user_choices, default='unknown')
    # Link username to user for graph builder compatibility
    df['user'] = df['username']

    # Privilege Level injection
    priv_choices = [
        'High',                 # Backdoors
        'Low',                  # DoS
        'High',                 # Exploits
        'Low',                  # Fuzzers
        'Medium',               # Generic
        'Low',                  # Reconnaissance
        'Medium',               # Shellcode
        'High',                 # Worms
        'High',                 # Analysis
        'System'                # Normal
    ]
    df['privilege_level'] = np.select(conditions, priv_choices, default='None')

    # Event Type injection
    event_choices = [
        'PROCESS_EXECUTION',    # Backdoors
        'NETWORK_CONNECTION',   # DoS
        'FILE_ACCESS',          # Exploits
        'PROCESS_EXECUTION',    # Fuzzers
        'NETWORK_CONNECTION',   # Generic
        'NETWORK_CONNECTION',   # Reconnaissance
        'PROCESS_EXECUTION',    # Shellcode
        'FILE_ACCESS',          # Worms
        'FILE_ACCESS',          # Analysis
        'NETWORK_CONNECTION'    # Normal
    ]
    df['event_type'] = np.select(conditions, event_choices, default='OTHER')

    return df

def clean_and_prepare_xdr_dataset(csv_path: str = None, df: pd.DataFrame = None, scaler=None) -> pd.DataFrame:
    """
    Follows the 3-step cleaning architecture:
    1. 'Trigger' Clean (before synthesis)
    2. Synthesize XDR Telemetry
    3. 'Formatting' Clean (after synthesis)
    """
    if df is None:
        if csv_path is None:
            raise ValueError("Must provide either csv_path or df")
        df = pd.read_csv(csv_path)
    
    # --- BEFORE SYNTHESIS ---
    # 1. Strip the invisible whitespaces so our synthesis conditions actually trigger
    if 'attack_cat' in df.columns:
        df['attack_cat'] = df['attack_cat'].fillna('Normal').astype(str).str.strip()
    else:
        df['attack_cat'] = 'Normal'
        
    # Standardize label if exists
    if 'label' in df.columns:
        df['label'] = df['label'].apply(lambda x: 1 if str(x).lower() in ['1', 'true', 'attack', 'malicious'] else 0)
    else:
        df['label'] = (df['attack_cat'] != 'Normal').astype(int)
        
    # --- SYNTHESIS ---
    # 2. Inject the files and processes based on the clean attack_cat
    df = synthesize_xdr_telemetry(df)
    
    # --- AFTER SYNTHESIS ---
    # 3. Now that the 'process' and 'file' columns exist, convert them to numbers
    categorical_cols = ['proto', 'state', 'process', 'username', 'privilege_level', 'event_type']
    for col in categorical_cols:
        if col in df.columns:
            df[col + '_encoded'] = df[col].astype('category').cat.codes
        else:
            df[col + '_encoded'] = 0
            
    # 4. Scale everything at the very end
    # Exclude non-numeric and encoded categorical features from scaling
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['label', 'id', 'srcip', 'dstip', 'timestamp'] + [c + '_encoded' for c in categorical_cols]
    numerical_features = [col for col in all_numeric if col not in exclude_cols]
    
    if len(numerical_features) > 0:
        if scaler is None:
            scaler = StandardScaler()
            df[numerical_features] = scaler.fit_transform(df[numerical_features].fillna(0))
        else:
            df[numerical_features] = scaler.transform(df[numerical_features].fillna(0))
            
    return df

# Alias for backwards compatibility with train_gat.py / graph_builder.py
def clean_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    return clean_and_prepare_xdr_dataset(df=df)

if __name__ == "__main__":
    import os
    input_file = r"c:\Users\Atharv Chaudhari\Desktop\Rohan\archive\UNSW_NB15_training-set.csv"
    output_file = r"c:\Users\Atharv Chaudhari\Desktop\Rohan\layer 2\layer 2\immunex_final_dataset.csv"
    
    if not os.path.exists(input_file):
        print(f"Could not find input file: {input_file}")
    else:
        print(f"Loading raw dataset from {input_file}...")
        try:
            df_processed = clean_and_prepare_xdr_dataset(csv_path=input_file)
            df_processed.to_csv(output_file, index=False)
            print(f"Successfully processed {len(df_processed)} events.")
            print(f"Saved synthesized endpoint XDR dataset to: {output_file}")
            
            # Print a quick preview of the synthesized columns
            synth_cols = ['attack_cat', 'process', 'file', 'username', 'event_type', 
                          'username_encoded', 'process_encoded']
            print("\nPreview of synthesized endpoint data:")
            print(df_processed[synth_cols].head())
        except Exception as e:
            print(f"Error during dataset preparation: {e}")
