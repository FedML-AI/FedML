import hashlib
from base64 import a85decode, a85encode

from ecies.utils import aes_decrypt, aes_encrypt
from nacl.public import Box, PrivateKey, PublicKey


def export_public_key(private_key_hex: str) -> bytes:
    """
    Export the public key for a contract join request.

    Args:
        private_key_hex (str): Hex string representing the private key.

    Returns:
        bytes: 32 bytes representing the public key.
    """

    def _hex_to_bytes(hex: str) -> bytes:
        """
        Convert a hex string to bytes.

        Args:
            hex (str): Hex string.

        Returns:
            bytes: Bytes representation of the hex string.
        """
        return bytes.fromhex(hex[2:] if hex[:2] == "0x" else hex)

    return bytes(PrivateKey(_hex_to_bytes(private_key_hex)).public_key)


def encrypt_nacl(public_key: bytes, data: bytes) -> bytes:
    """
    Encrypt data using NaCl box compatible with MetaMask.
    For implementation used in MetaMask look into: https://github.com/MetaMask/eth-sig-util

    Args:
        public_key (bytes): Public key of the recipient.
        data (bytes): Message data to be encrypted.

    Returns:
        bytes: Encrypted data.
    """
    emph_key = PrivateKey.generate()
    enc_box = Box(emph_key, PublicKey(public_key))
    # Encryption is required to work with MetaMask decryption (requires utf8)
    data = a85encode(data)
    ciphertext = enc_box.encrypt(data)
    return bytes(emph_key.public_key) + ciphertext


def decrypt_nacl(private_key: bytes, data: bytes) -> bytes:
    """Decryption function using NaCl box compatible with MetaMask
    For implementation used in MetaMask look into: https://github.com/MetaMask/eth-sig-util

    Args:
        private_key: private key to decrypt with
        data: encrypted message data

    Returns:
        decrypted data
    """
    emph_key, ciphertext = data[:32], data[32:]
    box = Box(PrivateKey(private_key), PublicKey(emph_key))
    return a85decode(box.decrypt(ciphertext))


def get_current_secret(secret: bytes, entry_key_turn: int, key_turn: int) -> bytes:
    """
    Calculate the shared secret at the current state.

    Args:
        secret (bytes): Initial secret.
        entry_key_turn (int): Entry key turn.
        key_turn (int): Key turn.

    Returns:
        bytes: The calculated shared secret.
    """
    for _ in range(entry_key_turn, key_turn):
        secret = hashlib.sha256(secret).digest()
    return secret


def encrypt(key: bytes, plain_text: bytes) -> bytes:
    return aes_encrypt(key, plain_text)


def decrypt(key: bytes, cipher_text: bytes) -> bytes:
    return aes_decrypt(key, cipher_text)
