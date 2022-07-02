package ai.fedml.edgedemo;

import android.content.Context;
import android.content.res.Resources;
import android.os.Handler;
import android.os.Looper;
import android.widget.Toast;

public class ToastUtils {
    private static final Handler sMainHandler = new Handler(Looper.getMainLooper());

    /**
     * Show Toast
     *
     * @param context context
     * @param text    文本
     */
    public static void show(final Context context, CharSequence text) {
        Runnable toastRunnable = () -> {
            if (text == null || text.equals("")) return;
            // 如果显示的文字超过了10个就显示长吐司，否则显示短吐司
            int duration = Toast.LENGTH_SHORT;
            if (text.length() > 20) {
                duration = Toast.LENGTH_LONG;
            }
            Toast.makeText(context, text, duration).show();
        };
        if (Looper.getMainLooper() == Looper.myLooper()) {
            toastRunnable.run();
        } else {
            sMainHandler.post(toastRunnable);
        }

    }

    /**
     * Show Toast
     *
     * @param context context
     * @param id      string resource id
     */
    public static void show(final Context context, int id) {
        try {
            // string resource id
            show(context, context.getResources().getText(id));
        } catch (Resources.NotFoundException ignored) {
            show(context, String.valueOf(id));
        }
    }
}
