# utils.py
import socket
import pickle
import struct

def send_data(conn, data):
    """Serializa (pickle) e envia dados prefixados com seu tamanho."""
    try:
        serialized_data = pickle.dumps(data)
        # 'L' é um unsigned long (4 bytes)
        msg_len = struct.pack('<L', len(serialized_data))
        conn.sendall(msg_len + serialized_data)
    except Exception as e:
        print(f"[ERRO ao Enviar] {e}")

def receive_data(conn):
    """Lê o prefixo de tamanho e depois recebe a quantidade exata de dados."""
    try:
        # Lê os primeiros 4 bytes para saber o tamanho da mensagem
        raw_msg_len = conn.recv(4)
        if not raw_msg_len:
            return None

        msg_len = struct.unpack('<L', raw_msg_len)[0]

        # Agora lê exatamente msg_len bytes
        data = b''
        while len(data) < msg_len:
            chunk = conn.recv(min(msg_len - len(data), 4096)) # Lê em pedaços de 4KB
            if not chunk:
                return None # Conexão fechada
            data += chunk

        return pickle.loads(data)
    except Exception as e:
        print(f"[ERRO ao Receber] {e}")
        return None
