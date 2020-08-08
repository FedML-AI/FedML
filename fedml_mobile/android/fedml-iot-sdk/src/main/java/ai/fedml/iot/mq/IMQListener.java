package ai.fedml.iot.mq;

import org.fusesource.hawtbuf.Buffer;
import org.fusesource.hawtbuf.UTF8Buffer;

public interface IMQListener {
    void onConnected();

    void onDisconnected();

    void onFailure(Throwable throwable);

    void onPublish(UTF8Buffer utf8Buffer, Buffer buffer, Runnable runnable);
}
