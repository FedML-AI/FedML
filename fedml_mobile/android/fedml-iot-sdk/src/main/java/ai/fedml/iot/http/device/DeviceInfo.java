package ai.fedml.iot.http.device;

/**
 * Created by hechaoyang on 1/20/18.
 */

public class DeviceInfo {
    /**
     * IOT_UUID : 59a525e4764c7d5ed6c99fe1ba8c9d97
     * deviceType : 1
     * channelID : Vehicle800001
     * osType : 1
     * osVersion : 4.4.4
     * appVersion : 1.0.0
     * mac : 34: 80: B3: 58: 44: B8
     * deviceID : fe014e50314a39354b121175f0e63200
     * deviceBrand : Xiaomi
     * deviceManufacturer : Xiaomi
     * deviceProduct : 2014112
     * deviceHardware : qcom
     * deviceBoard : msm8916
     * screenSize : 1024, 768
     * networkType : 4G
     */
    private String IOT_UUID;
    private int deviceType;
    private String channelID;
    private int osType;
    private String osVersion;
    private String appVersion;
    private String mac;
    private String deviceID;
    private String deviceBrand;
    private String deviceManufacturer;
    private String deviceProduct;
    private String deviceHardware;
    private String deviceBoard;
    private String screenSize;
    private String networkType;
    private String bluetoothMac;

    public String getIOT_UUID() {
        return IOT_UUID;
    }

    public void setIOT_UUID(String IOT_UUID) {
        this.IOT_UUID = IOT_UUID;
    }

    public int getDeviceType() {
        return deviceType;
    }

    public void setDeviceType(int deviceType) {
        this.deviceType = deviceType;
    }

    public String getChannelID() {
        return channelID;
    }

    public void setChannelID(String channelID) {
        this.channelID = channelID;
    }

    public int getOsType() {
        return osType;
    }

    public void setOsType(int osType) {
        this.osType = osType;
    }

    public String getOsVersion() {
        return osVersion;
    }

    public void setOsVersion(String osVersion) {
        this.osVersion = osVersion;
    }

    public String getAppVersion() {
        return appVersion;
    }

    public void setAppVersion(String appVersion) {
        this.appVersion = appVersion;
    }

    public String getMac() {
        return mac;
    }

    public void setMac(String mac) {
        this.mac = mac;
    }

    public String getDeviceID() {
        return deviceID;
    }

    public void setDeviceID(String deviceID) {
        this.deviceID = deviceID;
    }

    public String getDeviceBrand() {
        return deviceBrand;
    }

    public void setDeviceBrand(String deviceBrand) {
        this.deviceBrand = deviceBrand;
    }

    public String getDeviceManufacturer() {
        return deviceManufacturer;
    }

    public void setDeviceManufacturer(String deviceManufacturer) {
        this.deviceManufacturer = deviceManufacturer;
    }

    public String getDeviceProduct() {
        return deviceProduct;
    }

    public void setDeviceProduct(String deviceProduct) {
        this.deviceProduct = deviceProduct;
    }

    public String getDeviceHardware() {
        return deviceHardware;
    }

    public void setDeviceHardware(String deviceHardware) {
        this.deviceHardware = deviceHardware;
    }

    public String getDeviceBoard() {
        return deviceBoard;
    }

    public void setDeviceBoard(String deviceBoard) {
        this.deviceBoard = deviceBoard;
    }

    public String getScreenSize() {
        return screenSize;
    }

    public void setScreenSize(String screenSize) {
        this.screenSize = screenSize;
    }

    public String getNetworkType() {
        return networkType;
    }

    public void setNetworkType(String networkType) {
        this.networkType = networkType;
    }

    public String getBluetoothMac() {
        return bluetoothMac;
    }

    public void setBluetoothMac(String bluetoothMac) {
        this.bluetoothMac = bluetoothMac;
    }


    @Override
    public String toString() {
        return "DeviceInfo{" +
                "IOT_UUID='" + IOT_UUID + '\'' +
                ", deviceType=" + deviceType +
                ", channelID='" + channelID + '\'' +
                ", osType=" + osType +
                ", osVersion='" + osVersion + '\'' +
                ", appVersion='" + appVersion + '\'' +
                ", mac='" + mac + '\'' +
                ", deviceID='" + deviceID + '\'' +
                ", deviceBrand='" + deviceBrand + '\'' +
                ", deviceManufacturer='" + deviceManufacturer + '\'' +
                ", deviceProduct='" + deviceProduct + '\'' +
                ", deviceHardware='" + deviceHardware + '\'' +
                ", deviceBoard='" + deviceBoard + '\'' +
                ", screenSize='" + screenSize + '\'' +
                ", networkType='" + networkType + '\'' +
                ", bluetoothMac='" + bluetoothMac + '\'' +
                '}';
    }
}
