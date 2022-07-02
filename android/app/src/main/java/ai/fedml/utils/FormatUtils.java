package ai.fedml.utils;

import android.content.Context;
import android.text.format.DateFormat;

import java.math.BigDecimal;
import java.util.Date;

import ai.fedml.edge.service.ContextHolder;

public class FormatUtils {
    private FormatUtils() {
    }

    /**
     * long类型转换为String类型
     *
     * @param milSecond currentTime要转换的long类型的时间
     * @return LongDateFormat
     */
    public static String longToString(long milSecond) {
        Context context = ContextHolder.getAppContext();
        return DateFormat.getLongDateFormat(context).format(new Date(milSecond));
    }

    /**
     * byte 转换成 kb、mb、gb、tb
     *
     * @param average 字节大小
     * @return 文件大小
     */
    public static String unitConversion(long average) {
        double temp = average;
        if (temp < 1024) {
            BigDecimal result1 = new BigDecimal(temp);
            return result1.setScale(2, BigDecimal.ROUND_HALF_UP).doubleValue() + "B";
        }
        temp = temp / 1024;
        if (temp < 1024) {
            BigDecimal result1 = new BigDecimal(temp);
            return result1.setScale(2, BigDecimal.ROUND_HALF_UP).doubleValue() + "KB";
        }

        temp = temp / 1024;
        if (temp < 1024) {
            BigDecimal result1 = new BigDecimal(temp);
            return result1.setScale(2, BigDecimal.ROUND_HALF_UP).doubleValue() + "MB";
        }
        temp = temp / 1024;
        if (temp < 1024) {
            BigDecimal result1 = new BigDecimal(temp);
            return result1.setScale(2, BigDecimal.ROUND_HALF_UP).doubleValue() + "GB";
        }
        temp = temp / 1024;
        if (temp < 1024) {
            BigDecimal result1 = new BigDecimal(temp);
            return result1.setScale(2, BigDecimal.ROUND_HALF_UP).doubleValue() + "TB";
        }
        return "0";
    }
}
