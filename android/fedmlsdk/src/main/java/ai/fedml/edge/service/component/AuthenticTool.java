package ai.fedml.edge.service.component;

import java.nio.charset.StandardCharsets;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.util.Formatter;
import java.util.Locale;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

/**
 * Authentic tool
 *
 * @author jiezongchang
 */
public class AuthenticTool {
    private static final String MAC_SHA256 = "HmacSHA256";
    private final Mac mSha256Mac;

    public AuthenticTool(final String token) {
        Mac sha256Mac;
        try {
            sha256Mac = Mac.getInstance(MAC_SHA256);
            SecretKeySpec secretKeySpec = new SecretKeySpec(token.getBytes(StandardCharsets.UTF_8), MAC_SHA256);
            sha256Mac.init(secretKeySpec);
        } catch (InvalidKeyException | NoSuchAlgorithmException e) {
            throw new RuntimeException("hmacSha256 init failed", e);
        }
        mSha256Mac = sha256Mac;
    }

    /**
     * generate the authentic code by access token
     *
     * @param content the data
     * @return authentic code
     */
    public String generateAuthCode(final String content) {
        byte[] bytes = mSha256Mac.doFinal(content.getBytes(StandardCharsets.UTF_8));
        return toHexString(bytes).toUpperCase(Locale.ROOT);
    }

    /**
     * verify that the auth code is correct
     *
     * @param authCode authentic code
     * @param content  the data
     * @return correct or not
     */
    public boolean validateAuthCode(final String authCode, final String content) {
        return authCode.equals(generateAuthCode(content));
    }

    private String toHexString(byte[] bytes) {
        Formatter formatter = new Formatter();
        for (byte b : bytes) {
            formatter.format("%02x", b);
        }
        return formatter.toString();
    }
}
