package ai.fedml.edge.utils;

import android.util.Base64;

import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;

public class AesUtil {
    private static final String AES = "AES";
    private static final String CIPHER_MODE = "AES/ECB/PKCS5Padding";

    public static String encrypt(String cleartext, String key) {
        try {
            byte[] rawKey = getRawKey(key.getBytes());
            byte[] result = encrypt(rawKey, cleartext.getBytes());
            return Base64.encodeToString(result, Base64.DEFAULT);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static String decrypt(String encrypted, String key) {
        try {
            byte[] rawKey = getRawKey(key.getBytes());
            byte[] enc = Base64.decode(encrypted, Base64.DEFAULT);
            byte[] result = decrypt(rawKey, enc);
            return new String(result);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private static byte[] getRawKey(byte[] key) throws Exception {
        SecretKeySpec skeySpec = new SecretKeySpec(key, AES);
        return skeySpec.getEncoded();
    }

    private static byte[] encrypt(byte[] rawKey, byte[] clear) throws Exception {
        SecretKeySpec skeySpec = new SecretKeySpec(rawKey, AES);
        Cipher cipher = Cipher.getInstance(CIPHER_MODE);
        cipher.init(Cipher.ENCRYPT_MODE, skeySpec);
        return cipher.doFinal(clear);
    }

    private static byte[] decrypt(byte[] rawKey, byte[] encrypted) throws Exception {
        SecretKeySpec skeySpec = new SecretKeySpec(rawKey, AES);
        Cipher cipher = Cipher.getInstance(CIPHER_MODE);
        cipher.init(Cipher.DECRYPT_MODE, skeySpec);
        return cipher.doFinal(encrypted);
    }

}
