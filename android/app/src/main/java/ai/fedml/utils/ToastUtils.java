package ai.fedml.utils;

import android.content.Context;
import android.content.res.Resources;
import android.os.Handler;
import android.os.Looper;
import android.widget.Toast;

import ai.fedml.edge.service.ContextHolder;

public class ToastUtils {
    private static final Handler sMainHandler = new Handler(Looper.getMainLooper());

    /**
     * display Toast
     *
     * @param text
     */
    public static void show(CharSequence text) {
        Runnable toastRunnable = () -> {
            Context context = ContextHolder.getAppContext();
            if (text == null || text.equals("")) return;
            // If the displayed text exceeds 10, display the long toast, otherwise display the short toast
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
     * Display Toast
     *
     * @param id If the correct string id is passed in, the corresponding string will be displayed, if not, an integer string will be displayed
     */
    public static void show(int id) {
        try {
            // if this is a resource id
            show(ContextHolder.getAppContext().getResources().getText(id));
        } catch (Resources.NotFoundException ignored) {
            // if this is an int type
            show(String.valueOf(id));
        }
    }
}
