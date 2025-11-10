# utils.py
import pickle
import struct
import torch

def serialize_gradients(grads):
    """Serializa uma lista de tensores gradientes em um único objeto de bytes."""
    return pickle.dumps(grads)

def deserialize_gradients(data):
    """Desserializa bytes para obter a lista de tensores gradientes."""
    return pickle.loads(data)

def send_data(conn, data):
    """Envia dados pela conexão, prefixados com o tamanho."""
    data_bytes = serialize_gradients(data)
    # Empacota o tamanho dos dados como um inteiro de 4 bytes (little-endian)
    size_prefix = struct.pack('<L', len(data_bytes))
    conn.sendall(size_prefix + data_bytes)

def receive_data(conn):
    """Recebe dados com o tamanho prefixado."""
    # Recebe o prefixo de 4 bytes (tamanho)
    size_prefix = conn.recv(4)
    if not size_prefix:
        return None

    data_size = struct.unpack('<L', size_prefix)[0]

    # Recebe o corpo dos dados
    data_chunks = []
    bytes_recd = 0
    while bytes_recd < data_size:
        # Pega o que for menor: o que falta ou um buffer padrão (4096)
        chunk = conn.recv(min(data_size - bytes_recd, 4096))
        if not chunk:
            raise RuntimeError("Conexão interrompida antes de receber todos os dados.")
        data_chunks.append(chunk)
        bytes_recd += len(chunk)

    data_bytes = b"".join(data_chunks)
    return deserialize_gradients(data_bytes)
