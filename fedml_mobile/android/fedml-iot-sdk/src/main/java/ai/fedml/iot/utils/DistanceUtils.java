package ai.fedml.iot.utils;

import android.location.Location;

public class DistanceUtils {

    /**
     * 计算两点间的直线距离
     *
     * @param sx 起点X坐标
     * @param sy 起点Y坐标
     * @param ex 终点X坐标
     * @param ey 终点Y坐标
     * @return
     */
    public static double lineDistance(double sx, double sy, double ex, double ey) {
        double dDist = Math.sqrt((ex - sx) * (ex - sx) + (ey - sy) * (ey - sy));
        return dDist;
    }

    /**
     * 计算两地理坐标球面距离（gcj02坐标系）
     *
     * @param lat1 纬度
     * @param lng1 经度
     * @param lat2 纬度
     * @param lng2 经度
     * @return 米
     */
    public static int geoSphereDistance(double lat1, double lng1, double lat2, double lng2) {
        float[] results = new float[1];
        Location.distanceBetween(lat1, lng1, lat2, lng2, results);
//        LogUtils.d("geoSphereDistance (" + lat1 + "," + lng1 + ") (" + lat2 + "," + lng2 + ") = " + (int)results[0]);
        return (int) results[0];
    }
}
