import os
import subprocess
import paramiko

# Configurazione
local_base_dir = os.path.dirname(os.path.abspath(__file__))
remote_user = "chiesa"
remote_host = "hssh1.di.unito.it"
remote_base_dir = "/scratch/Tesi-Borra"


def establish_ssh_connection():
    private_key_path = os.path.expanduser("/Users/giorgiochiesa/.ssh/id_ed25519")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key = paramiko.Ed25519Key.from_private_key_file(private_key_path)
    ssh.connect(remote_host, username=remote_user, pkey=key)
    print(f"Connessione SSH stabilita con {remote_host} come {remote_user}")
    ssh.close()

def sync_files():
    # Costruisce il comando scp
    private_key_path = os.path.expanduser("/Users/giorgiochiesa/.ssh/id_ed25519")  # Modifica il percorso se necessario
    scp_command = [
        "scp",
        "-i", private_key_path,  # usa la chiave privata per autenticazione
        "-r",  # ricorsivo per copiare directory
        local_base_dir + "/",  # aggiungi / per copiare il contenuto della cartella
        f"{remote_user}@{remote_host}:{remote_base_dir}/"
    ]
    print("Eseguo:", " ".join(scp_command))
    subprocess.run(scp_command, check=True)

if __name__ == "__main__":
    establish_ssh_connection()
    # sync_files()