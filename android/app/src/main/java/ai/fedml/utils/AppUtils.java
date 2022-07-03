package ai.fedml.utils;

import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Looper;
import android.view.View;
import android.widget.Toast;

import java.math.BigDecimal;


public class AppUtils {





    /**
     * Get local version number
     */
    public static int getVersionCode(Context mContext) {
        int versionCode = 0;
        try {
            //Get the software version number, corresponding to android:versionCode under AndroidManifest.xml
            versionCode = mContext.getPackageManager().
                    getPackageInfo(mContext.getPackageName(), 0).versionCode;
        } catch (PackageManager.NameNotFoundException e) {
            e.printStackTrace();
        }
        return versionCode;
    }

}
