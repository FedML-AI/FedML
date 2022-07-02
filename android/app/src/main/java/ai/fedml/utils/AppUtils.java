package ai.fedml.utils;

import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Looper;
import android.view.View;
import android.widget.Toast;

import java.math.BigDecimal;

/**
 * @创建者 xkai
 * @创建时间 2022/1/5 15:45
 * @描述
 */
public class AppUtils {





    /**
     * 获取本地版本号
     */
    public static int getVersionCode(Context mContext) {
        int versionCode = 0;
        try {
            //获取软件版本号，对应AndroidManifest.xml下android:versionCode
            versionCode = mContext.getPackageManager().
                    getPackageInfo(mContext.getPackageName(), 0).versionCode;
        } catch (PackageManager.NameNotFoundException e) {
            e.printStackTrace();
        }
        return versionCode;
    }

}
