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
     * 显示Toast
     *
     * @param text 文本
     */
    public static void show(CharSequence text) {
        Runnable toastRunnable = () -> {
            Context context = ContextHolder.getAppContext();
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
     * 显示Toast
     *
     * @param id 如果传入的是正确的string id就显示对应字符串,如果不是则显示一个整数的string
     */
    public static void show(int id) {
        try {
            // 如果这是一个资源id
            show(ContextHolder.getAppContext().getResources().getText(id));
        } catch (Resources.NotFoundException ignored) {
            // 如果这是一个int类型
            show(String.valueOf(id));
        }
    }
}
