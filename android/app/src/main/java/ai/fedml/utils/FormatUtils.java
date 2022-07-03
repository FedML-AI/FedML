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
     * Convert long type to String type
     *
     * @param milSecond currentTime time of type long to convert
     * @return LongDateFormat
     */
    public static String longToString(long milSecond) {
        Context context = ContextHolder.getAppContext();
        return DateFormat.getLongDateFormat(context).format(new Date(milSecond));
    }

    /**
     * byte convert to kb、mb、gb、tb
     *
     * @param average size in bytes
     * @return File size
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
