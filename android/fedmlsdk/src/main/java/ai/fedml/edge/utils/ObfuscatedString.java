package ai.fedml.edge.utils;

import java.nio.charset.StandardCharsets;
import java.util.Random;

import androidx.annotation.NonNull;

/**
 * This class is designed to be thread safe.
 *
 */
public class ObfuscatedString {
  
    /**
     * The obfuscated string.
     */
    private final long[] obfuscated;

    /**
     * Constructs an obfuscated string.
     *
     * @param obfuscated The obfuscated string.
     * @throws NullPointerException           If {@code obfuscated} is
     *                                        {@code null}.
     * @throws ArrayIndexOutOfBoundsException If the provided array does not
     *                                        contain at least one element.
     */
    public ObfuscatedString(final long[] obfuscated) {
        this.obfuscated = obfuscated.clone();
        this.obfuscated[0] = obfuscated[0];
    }

    /**
     * Returns the original string.
     */
    @NonNull
    @Override
    public String toString() {
        final int length = obfuscated.length;
        // The original UTF8 encoded string was probably not a multiple
        // of eight bytes long and is thus actually shorter than this array.
        final byte[] encoded = new byte[8 * (length - 1)];

        // Obtain the seed and initialize a new PRNG with it.
        final long seed = obfuscated[0];
        final Random prng = new Random(seed);
        // De-obfuscate.
        for (int i = 1; i < length; i++) {
            final long key = prng.nextLong();
            toBytes(obfuscated[i] ^ key, encoded, 8 * (i - 1));
        }

        final String decoded = new String(encoded, StandardCharsets.UTF_8);
        final int i = decoded.indexOf(0);
        return -1 == i ? decoded : decoded.substring(0, i);
    }

    /**
     * @param l     The long value to encode.
     * @param bytes The array which holds the encoded bytes upon return.
     * @param off   The offset of the bytes in the array.
     */
    private static void toBytes(long l, byte[] bytes, int off) {
        final int end = Math.min(bytes.length, off + 8);
        for (int i = off; i < end; i++) {
            bytes[i] = (byte) l;
            l >>= 8;
        }
    }
}
